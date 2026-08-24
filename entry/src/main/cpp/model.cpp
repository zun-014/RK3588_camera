// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "model.h"
#include "hilog/log.h"
#include <rawfile/raw_file_manager.h>
#include <memory>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <cmath>
#include <cstring>
#include "rknn/rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "postprocess.h"
#include "preprocess.h"
#include <sys/time.h>
const int GLOBAL_RESMGR = 0xFF00;
const char *MODEL_TAG = "[Model_Utils]";

// Facenet模型路径和参数
#define FACENET_MODEL_PATH "rawfile/model/usemodel/facenet.rknn"
#define FACENET_INPUT_SIZE 160  // Facenet输入尺寸
double __get_us_a(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }
// Facenet模型相关全局变量
static rknn_context facenet_ctx = 0;
static bool facenet_initialized = false;

// 人脸识别相关结构体
struct FaceRecognitionTask {
    std::string face_path;
    std::string sandbox_dir;
    NativeResourceManager* resource_manager;
};

struct FaceRecognitionResult {
    float similarity;
    bool is_same_person;
    bool has_result;
    std::string processed_file;
    std::string matched_reference_path; // 匹配的参考人脸图片路径
};

// 人脸识别相关全局变量
static std::vector<std::vector<float>> facenet_reference_features_; // 存储多个参考特征向量
static std::vector<std::string> facenet_reference_paths_; // 存储参考特征对应的图片路径
static std::queue<FaceRecognitionTask> face_recognition_queue;
static std::mutex queue_mutex;
static std::mutex result_mutex;
static std::atomic<bool> worker_thread_running(false);
static std::thread worker_thread;
static FaceRecognitionResult latest_recognition_result = {0.0f, false, false, "", ""};
static bool numbered_images_processed = false; // 标记数字命名图片是否已处理过
static bool is_generate_face_feature = true; // 控制模式：true为特征录入模式，false为识别模式


/**
 * 加载并保存模型文件
 * @param mNativeResMgr 资源管理器指针
 * @param rawfile_path 原始文件路径
 * @param save_path 保存路径
 * @return 成功返回true，失败返回false
 */
bool load_and_save_model(NativeResourceManager* mNativeResMgr, const std::string& rawfile_path, const std::string& save_path) {
    // 获取模型数据
    RawFile *rawFile = OH_ResourceManager_OpenRawFile(mNativeResMgr, rawfile_path.c_str());
    if (rawFile == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MODEL_TAG, "Failed to open model file: %{public}s", rawfile_path.c_str());
        return false;
    }
    
    // 获取模型数据大小并读取
    long modelSize = OH_ResourceManager_GetRawFileSize(rawFile);
    std::unique_ptr<uint8_t[]> modelData = std::make_unique<uint8_t[]>(modelSize);
    int readResult = OH_ResourceManager_ReadRawFile(rawFile, modelData.get(), modelSize);
    OH_ResourceManager_CloseRawFile(rawFile);
    
    if (readResult != modelSize) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MODEL_TAG, "Failed to read model data from: %{public}s", rawfile_path.c_str());
        return false;
    }
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG, "Model data loaded successfully from %{public}s, size: %{public}ld", rawfile_path.c_str(), modelSize);
    
    // 将模型数据保存到文件
    FILE *file = fopen(save_path.c_str(), "wb");
    if (file == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MODEL_TAG, "Failed to create model file: %{public}s", save_path.c_str());
        return false;
    }
    
    size_t written = fwrite(modelData.get(), 1, modelSize, file);
    fclose(file);
    
    if (written != modelSize) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MODEL_TAG, "Failed to write model data to file: %{public}s", save_path.c_str());
        return false;
    }
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG, "Model file saved successfully to: %{public}s", save_path.c_str());
    return true;
}

/**
 * 将量化后的数据转换为浮点数
 * @param qnt 量化后的整数值
 * @param zp 零点偏移
 * @param scale 缩放因子
 * @return 转换后的浮点数值
 */
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}

/**
 * 计算两个向量的余弦相似度
 * @param vec1 第一个特征向量
 * @param vec2 第二个特征向量
 * @return 余弦相似度值（0.0-1.0）
 */
static float calculate_cosine_similarity(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    if (vec1.size() != vec2.size() || vec1.empty()) {
        return 0.0f;
    }
    
    float dot_product = 0.0f;
    for (size_t i = 0; i < vec1.size(); i++) {
        dot_product += vec1[i] * vec2[i];
    }
    
    // 由于向量已经归一化，点积即为余弦相似度
    // 如果相似度为负数，则设为0（表示完全不相似）
    return dot_product < 0.0f ? 0.0f : dot_product;
}

/**
 * Facenet推理函数
 * @param face_image 输入的人脸图像
 * @param feature_vector 输出的特征向量
 * @param mNativeResMgr 资源管理器指针
 * @return 成功返回0，失败返回-1
 */
int facenet_inference(const cv::Mat& face_image, std::vector<float>& feature_vector, NativeResourceManager* mNativeResMgr) {
    auto facenet_total_start = std::chrono::steady_clock::now();
    // 初始化Facenet模型（如果尚未初始化）
    if (!facenet_initialized) {
        RawFile *facenetRawFile = OH_ResourceManager_OpenRawFile(mNativeResMgr, FACENET_MODEL_PATH);
        if (facenetRawFile == nullptr) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MODEL_TAG, "Failed to open Facenet model file");
            return -1;
        }
        
        long facenetModelSize = OH_ResourceManager_GetRawFileSize(facenetRawFile);
        std::unique_ptr<uint8_t[]> facenetModelData = std::make_unique<uint8_t[]>(facenetModelSize);
        int readResult = OH_ResourceManager_ReadRawFile(facenetRawFile, facenetModelData.get(), facenetModelSize);
        OH_ResourceManager_CloseRawFile(facenetRawFile);
        
        if (readResult != facenetModelSize) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MODEL_TAG, "Failed to read Facenet model data");
            return -1;
        }
        
        int ret = rknn_init(&facenet_ctx, facenetModelData.get(), facenetModelSize, 0, NULL);
        if (ret < 0) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MODEL_TAG, "Facenet model initialization failed");
            return -1;
        }
        
        facenet_initialized = true;
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG, "Facenet model initialized successfully");
    }
    
    // 预处理人脸图像
    cv::Mat resized_face;
    cv::resize(face_image, resized_face, cv::Size(FACENET_INPUT_SIZE, FACENET_INPUT_SIZE));

    cv::Mat rgb_face;
    if (face_image.channels() == 3) {
        cv::cvtColor(resized_face, rgb_face, cv::COLOR_BGR2RGB);
    } else {
        rgb_face = resized_face;
    }
    
    // 准备输入数据
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = FACENET_INPUT_SIZE * FACENET_INPUT_SIZE * 3;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = rgb_face.data;
    
    // 设置输入
    int ret = rknn_inputs_set(facenet_ctx, 1, inputs);
    if (ret != RKNN_SUCC) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MODEL_TAG, "rknn_inputs_set failed for Facenet! ret=%{public}d", ret);
        return -1;
    }
    float inference_time;
    struct timeval start_time,stop_time;
    gettimeofday(&start_time, NULL);
    // 执行推理
    ret = rknn_run(facenet_ctx, NULL);
    if (ret != RKNN_SUCC) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MODEL_TAG, "rknn_run failed for Facenet! ret=%{public}d", ret);
        return -1;
    }
    gettimeofday(&stop_time, NULL);
    inference_time = (__get_us_a(stop_time) - __get_us_a(start_time)) / 1000;

    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG, "facenet Inference time: %{public}f ms", inference_time);
    
    // 获取输出
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].index = 0;
    outputs[0].want_float = 0;
    
    ret = rknn_outputs_get(facenet_ctx, 1, outputs, NULL);
    if (ret != RKNN_SUCC) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MODEL_TAG, "rknn_outputs_get failed for Facenet! ret=%{public}d", ret);
        return -1;
    }
    
    // 查询输出属性
    rknn_tensor_attr output_attr;
    output_attr.index = 0;
    ret = rknn_query(facenet_ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MODEL_TAG, "Failed to query Facenet output attr");
        rknn_outputs_release(facenet_ctx, 1, outputs);
        return -1;
    }
    
    // 反量化并归一化
    feature_vector.resize(128);
    int8_t* output_data = (int8_t*)outputs[0].buf;
    
    for (int i = 0; i < 128; i++) {
        feature_vector[i] = deqnt_affine_to_f32(output_data[i], output_attr.zp, output_attr.scale);
    }
    
    // L2归一化
    float sum = 0;
    for (int i = 0; i < 128; i++) {
        sum += feature_vector[i] * feature_vector[i];
    }
    float norm = std::sqrt(sum);
    for (int i = 0; i < 128; i++) {
        feature_vector[i] /= norm;
    }
    
    rknn_outputs_release(facenet_ctx, 1, outputs);
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG, "Facenet inference completed successfully");
    auto facenet_total_end = std::chrono::steady_clock::now();
    auto facenet_total_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            facenet_total_end - facenet_total_start
        ).count();
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG,
        "Facenet TOTAL time: %{public}ld ms", facenet_total_ms);
    return 0;
}

/**
 * 异步人脸识别工作线程函数
 */
static void face_recognition_worker() {
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG, "Face recognition worker thread started");
    
    while (worker_thread_running.load()) {
        FaceRecognitionTask task;
        bool has_task = false;
        
        // 从队列中获取任务
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (!face_recognition_queue.empty()) {
                task = face_recognition_queue.front();
                face_recognition_queue.pop();
                has_task = true;
            }
        }
        
        if (has_task) {
            auto worker_start = std::chrono::steady_clock::now();
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG, "Processing face recognition task: %{public}s", task.face_path.c_str());
            
            // 读取预处理后的人脸图像
            cv::Mat face_image = cv::imread(task.face_path, cv::IMREAD_COLOR);
            if (!face_image.data) {
                OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MODEL_TAG, "Failed to read face image: %{public}s", task.face_path.c_str());
                continue;
            }
            
            std::vector<float> current_feature;
            int facenet_ret = facenet_inference(face_image, current_feature, task.resource_manager);
            
            if (facenet_ret == 0 && !facenet_reference_features_.empty()) {
                    float max_similarity = 0.0f;
                    std::string matched_path = "";
                    for (size_t i = 0; i < facenet_reference_features_.size(); ++i) {
                        float similarity = calculate_cosine_similarity(current_feature, facenet_reference_features_[i]);
                        if (similarity > max_similarity) {
                            max_similarity = similarity;
                            if (i < facenet_reference_paths_.size()) {
                                matched_path = facenet_reference_paths_[i];
                            }
                        }
                    }
                    float similarity = max_similarity;
                bool is_same_person = similarity > 0.6f;
                
                // 更新结果
                {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    latest_recognition_result.similarity = similarity;
                    latest_recognition_result.is_same_person = is_same_person;
                    latest_recognition_result.has_result = true;
                    latest_recognition_result.processed_file = task.face_path;
                    latest_recognition_result.matched_reference_path = matched_path;
                }
                
                OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG, "Face recognition completed: %{public}s (similarity: %{public}f)", 
                           is_same_person ? "Same person" : "Different person", similarity);
            } else {
                OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MODEL_TAG, "Face recognition failed for: %{public}s", task.face_path.c_str());
            }
            auto worker_end = std::chrono::steady_clock::now();
            auto worker_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    worker_end - worker_start
                ).count();
            
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG,
                "FaceRecognition WORKER TOTAL time: %{public}ld ms", worker_ms);
        } else {
            // 没有任务时短暂休眠
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG, "Face recognition worker thread stopped");
}

/**
 * 启动异步人脸识别工作线程
 */
void start_face_recognition_worker() {
    if (!worker_thread_running.load()) {
        worker_thread_running.store(true);
        worker_thread = std::thread(face_recognition_worker);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG, "Face recognition worker thread started");
    }
}

/**
 * 停止异步人脸识别工作线程
 */
void stop_face_recognition_worker() {
    if (worker_thread_running.load()) {
        worker_thread_running.store(false);
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG, "Face recognition worker thread stopped");
    }
}

/**
 * 添加人脸识别任务到队列
 * @param face_path 人脸图像文件路径
 * @param sandbox_dir 沙箱目录路径
 * @param resource_manager 资源管理器指针
 */
void add_face_recognition_task(const std::string& face_path, const std::string& sandbox_dir, NativeResourceManager* resource_manager) {
    FaceRecognitionTask task;
    task.face_path = face_path;
    task.sandbox_dir = sandbox_dir;
    task.resource_manager = resource_manager;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        face_recognition_queue.push(task);
    }
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG, "Added face recognition task: %{public}s", face_path.c_str());
}

/**
 * 获取最新的人脸识别结果
 * @param similarity 相似度引用
 * @param is_same_person 是否为同一人引用
 * @param has_result 是否有结果引用
 * @param processed_file 处理的文件路径引用
 * @param matched_reference_path 匹配的参考图片路径引用
 * @return 有新结果返回true，无新结果返回false
 */
bool get_latest_face_recognition_result(float& similarity, bool& is_same_person, bool& has_result, std::string& processed_file, std::string& matched_reference_path) {
    std::lock_guard<std::mutex> lock(result_mutex);
    if (latest_recognition_result.has_result) {
        similarity = latest_recognition_result.similarity;
        is_same_person = latest_recognition_result.is_same_person;
        has_result = latest_recognition_result.has_result;
        processed_file = latest_recognition_result.processed_file;
        matched_reference_path = latest_recognition_result.matched_reference_path;
        
        // 重置结果状态
        latest_recognition_result.has_result = false;
        return true;
    }
    return false;
}

/**
 * 添加参考人脸特征向量
 * @param feature_vector 特征向量
 * @param image_path 对应的图像路径
 */
void add_reference_face_feature(const std::vector<float>& feature_vector, const std::string& image_path) {
    facenet_reference_features_.push_back(feature_vector);
    facenet_reference_paths_.push_back(image_path);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG, "Added reference face feature: %{public}s", image_path.c_str());
}

/**
 * 清空所有参考人脸特征向量
 */
void clear_reference_face_features() {
    facenet_reference_features_.clear();
    facenet_reference_paths_.clear();
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG, "Cleared all reference face features");
}

/**
 * 获取参考人脸特征向量的数量
 * @return 特征向量数量
 */
size_t get_reference_face_count() {
    return facenet_reference_features_.size();
}

/**
 * 设置人脸识别模式
 * @param generate_mode true为特征生成模式，false为识别模式
 */
void set_face_recognition_mode(bool generate_mode) {
    is_generate_face_feature = generate_mode;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG, "Face recognition mode set to: %{public}s", 
                generate_mode ? "Feature generation" : "Recognition");
}

/**
 * 获取当前人脸识别模式
 * @return true为特征生成模式，false为识别模式
 */
bool get_face_recognition_mode() {
    return is_generate_face_feature;
}

/**
 * 设置数字命名图片的处理状态
 * @param processed 是否已处理
 */
void set_numbered_images_processed(bool processed) {
    numbered_images_processed = processed;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MODEL_TAG, "Numbered images processed status set to: %{public}s", 
                processed ? "true" : "false");
}

bool get_numbered_images_processed() {
    return numbered_images_processed;
}
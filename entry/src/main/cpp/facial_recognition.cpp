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

#include "napi/native_api.h"
#include <rawfile/raw_file.h>
#include <rawfile/raw_file_manager.h>
#include "hilog/log.h"
#include "preprocess.h"
#include "postprocess.h"
#include "rknn/rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "face_crop_utils.h"
#include "Common.hpp"
#include "model.h"
#include <string.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <map>
#include <vector>
#include <algorithm>
#include <dirent.h>                 // 添加目录操作相关头文件
#include <sys/stat.h>               // 添加stat结构体和函数声明
#include <unistd.h>                 // 添加unlink函数声明
#include <chrono>                   // 添加时间相关头文件
#include <thread>                   // 添加线程支持
#include <mutex>                    // 添加互斥锁支持
#include <atomic>                   // 添加原子操作支持
#include <queue>                    // 添加队列支持
// RKNN模型数据结构
struct RknnModelData {
    std::unique_ptr<uint8_t[]> data;
    long size;
};

// RKNN模型信息结构
struct RknnModelInfo {
    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attr;
    rknn_tensor_attr* output_attr;
    int width;
    int height;
    int channel;
};

const int GLOBAL_RESMGR = 0xFF00; 
const char *TAG = "[Sample_rawfile]";
const char *TAG_Face = "[Face_recongition]";

static RknnModelData g_yolo_model_data;     // 保存模型字节，保证内存长期有效
static RknnModelInfo g_yolo_model_info;

bool load_data_flag = false;
bool Rknn_init_flag = false;


/**
 * 对图像进行letterbox处理，保持宽高比的同时调整到目标尺寸
 * @param image 输入的原始图像
 * @param padded_image 输出的填充后图像
 * @param pads 填充信息结构体，记录各边的填充像素数
 * @param scale 缩放比例
 * @param target_size 目标图像尺寸
 * @param pad_color 填充颜色
 */
void letterbox(const cv::Mat &image, cv::Mat &padded_image, BOX_RECT &pads, const float scale, const cv::Size &target_size, const cv::Scalar &pad_color)
{
    // 调整图像大小
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(), scale, scale);

    // 计算填充大小
    int pad_width = target_size.width - resized_image.cols;
    int pad_height = target_size.height - resized_image.rows;

    pads.left = pad_width / 2;
    pads.right = pad_width - pads.left;
    pads.top = pad_height / 2;
    pads.bottom = pad_height - pads.top;

    // 在图像周围添加填充
    cv::copyMakeBorder(resized_image, padded_image, pads.top, pads.bottom, pads.left, pads.right, cv::BORDER_CONSTANT, pad_color);
}

/**
 * 将timeval结构转换为微秒数
 * @param t timeval时间结构体
 * @return 对应的微秒数
 */
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

/**
 * 清空所有face*_*_preprocessed.jpg文件
 * @param output_dir 输出目录路径
 */
static void cleanup_all_face_preprocessed_files(const std::string& output_dir) {
    try {
        // 使用传统的C函数遍历目录
        DIR* dir = opendir(output_dir.c_str());
        if (dir == nullptr) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, 
                       "Failed to open directory for cleanup: %{public}s", output_dir.c_str());
            return;
        }
        
        struct dirent* entry;
        std::vector<std::string> files_to_delete;
        
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            
            // 检查是否是face*_*_preprocessed.jpg或face*_*.jpg文件
            if (filename.find("face") == 0 && 
                (filename.find("_preprocessed.jpg") != std::string::npos || 
                 (filename.find(".jpg") != std::string::npos && 
                  filename.find("_") != std::string::npos))) {
                std::string full_path = output_dir + "/" + filename;
                
                // 检查是否是普通文件
                struct stat file_stat;
                if (stat(full_path.c_str(), &file_stat) == 0 && S_ISREG(file_stat.st_mode)) {
                    files_to_delete.push_back(full_path);
                }
            }
        }
        closedir(dir);
        
        // 删除找到的文件
        for (const std::string& file_path : files_to_delete) {
            if (unlink(file_path.c_str()) == 0) {
                OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, 
                           "Deleted face file: %{public}s", file_path.c_str());
            } else {
                OH_LOG_Print(LOG_APP, LOG_WARN, GLOBAL_RESMGR, TAG, 
                           "Failed to delete face file: %{public}s", file_path.c_str());
            }
        }
        
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, 
                   "Cleanup completed: %{public}d face files processed", 
                   (int)files_to_delete.size());
        
    } catch (const std::exception& e) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, 
                   "Error during face files cleanup: %{public}s", e.what());
    }
}

/**
 * 加载RKNN模型数据
 * @param sandboxDir 沙箱目录路径
 * @param mNativeResMgr 原生资源管理器指针
 * @param modelData 模型数据结构体引用，用于存储加载的模型数据
 * @return 加载成功返回true，失败返回false
 */
static bool LoadRknnModelData(const std::string& sandboxDir, NativeResourceManager* mNativeResMgr, RknnModelData& modelData) {
    // 构建沙箱中模型文件的路径
    std::string modelFilePath = sandboxDir + "/yolov5_model.rknn";
    bool modelLoaded = false;
    
    // 尝试从沙箱目录读取模型文件
    FILE *file = fopen(modelFilePath.c_str(), "rb");
    if (file != nullptr) {
        // 获取文件大小
        fseek(file, 0, SEEK_END);
        modelData.size = ftell(file);
        fseek(file, 0, SEEK_SET);
        
        if (modelData.size > 0) {
            // 读取模型数据
            modelData.data = std::make_unique<uint8_t[]>(modelData.size);
            size_t readSize = fread(modelData.data.get(), 1, modelData.size, file);
            fclose(file);
            
            if (readSize == modelData.size) {
                OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Model loaded from sandbox: %{public}s, size: %{public}ld", modelFilePath.c_str(), modelData.size);
                modelLoaded = true;
            } else {
                OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to read model from sandbox, expected size: %{public}ld, read size: %{public}ld", modelData.size, readSize);
            }
        } else {
            fclose(file);
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Model file in sandbox has invalid size: %{public}ld", modelData.size);
        }
    } else {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Model file not found in sandbox, will load from resource: %{public}s", modelFilePath.c_str());
    }
    
    // 如果从沙箱加载失败，则从ResourceManager加载
    if (!modelLoaded) {
        // 获取模型数据
        RawFile *rawFile = OH_ResourceManager_OpenRawFile(mNativeResMgr, "rawfile/model/usemodel/yolov5.rknn");
        if (rawFile == nullptr) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to open model file");
            return false;
        }
        
        // 获取模型数据大小并读取
        modelData.size = OH_ResourceManager_GetRawFileSize(rawFile);
        modelData.data = std::make_unique<uint8_t[]>(modelData.size);
        int readResult = OH_ResourceManager_ReadRawFile(rawFile, modelData.data.get(), modelData.size);
        OH_ResourceManager_CloseRawFile(rawFile);
        
        if (readResult != modelData.size) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to read model data");
            return false;
        }
        
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Model data loaded from resource successfully, size: %{public}ld", modelData.size);
    }
    
    return true;
}

static bool ValidateRknnInitParams(const RknnModelData& modelData, RknnModelInfo& modelInfo) {
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, 
                 "=== RKNN初始化参数验证 ===");
    
    // 1. 验证模型数据指针
    uint8_t* data_ptr = modelData.data.get();
    if (!data_ptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, 
                     "模型数据指针为NULL");
        return false;
    }
    
    // 安全地打印指针地址（避免隐私保护）
    char ptr_str[32];
    snprintf(ptr_str, sizeof(ptr_str), "%p", data_ptr);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, 
                 "模型数据指针: %{public}s", ptr_str);
    
    // 2. 验证指针是否可读
    // 尝试读取前4字节来验证指针有效性
    volatile uint32_t test_read = 0;
  
        test_read = *reinterpret_cast<uint32_t*>(data_ptr);
        char magic_str[16];
        snprintf(magic_str, sizeof(magic_str), "0x%08X", test_read);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, 
                     "指针可读，前4字节: %{public}s", magic_str);

    
    // 3. 验证内存对齐
    uintptr_t addr = reinterpret_cast<uintptr_t>(data_ptr);
    bool is_4_aligned = (addr % 4 == 0);
    bool is_8_aligned = (addr % 8 == 0);
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, 
                 "内存对齐 - 4字节: %{public}s, 8字节: %{public}s",
                 is_4_aligned ? "是" : "否",
                 is_8_aligned ? "是" : "否");
    
    // RKNN通常需要4字节对齐
    if (!is_4_aligned) {
        OH_LOG_Print(LOG_APP, LOG_WARN, GLOBAL_RESMGR, TAG, 
                     "警告: 模型数据未4字节对齐，可能导致初始化失败");
    }
    
    // 4. 验证模型大小
    if (modelData.size == 0) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, 
                     "模型大小为0");
        return false;
    }
    
    // 判断大小是否合理
    const size_t MIN_RKNN_SIZE = 1024;      // 最小1KB
    const size_t MAX_RKNN_SIZE = 100 * 1024 * 1024; // 最大100MB
    const size_t EXPECTED_SIZE = 3752828;   // 期望的大小
    
    if (modelData.size < MIN_RKNN_SIZE) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, 
                     "模型大小太小: %zu 字节", modelData.size);
        return false;
    }
    
    if (modelData.size > MAX_RKNN_SIZE) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, 
                     "模型大小太大: %zu 字节", modelData.size);
        return false;
    }
    
    char size_str[32];
    snprintf(size_str, sizeof(size_str), "%zu", modelData.size);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, 
                 "模型大小: %{public}s 字节", size_str);
    
    // 检查是否与期望大小匹配
    if (modelData.size == EXPECTED_SIZE) {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, 
                     "模型大小与期望值匹配 (3752828 字节)");
    } else {
        OH_LOG_Print(LOG_APP, LOG_WARN, GLOBAL_RESMGR, TAG, 
                     "模型大小与期望值不匹配，期望: 3752828，实际: %zu", 
                     modelData.size);
    }
    
    // 5. 验证模型头格式
    uint32_t magic = *reinterpret_cast<uint32_t*>(data_ptr);
    char magic_hex[16];
    snprintf(magic_hex, sizeof(magic_hex), "0x%08X", magic);
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, 
                 "模型魔数: %{public}s", magic_hex);
    
    bool valid_magic = false;
    if (magic == 0x4E4E4B52) {          // "RKNN" 小端序
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, 
                     "RKNN v1.x 格式");
        valid_magic = true;
    } else if (magic == 0x01004B52) {   // "RK\x01\x00"
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, 
                     "RKNN v2.x 格式");
        valid_magic = true;
    } else if (magic == 0x4E4B4E52) {   // "RNNK" 可能的大端序
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, 
                     "RKNN 大端序格式");
        valid_magic = true;
    } else {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, 
                     "无效的RKNN魔数");
        
        // 打印前16字节用于调试
        char hex_dump[64] = {0};
        for (int i = 0; i < 16 && i < modelData.size; i++) {
            char byte_str[4];
            snprintf(byte_str, sizeof(byte_str), "%02X ", data_ptr[i]);
            strcat(hex_dump, byte_str);
        }
        OH_LOG_Print(LOG_APP, LOG_DEBUG, GLOBAL_RESMGR, TAG, 
                     "前16字节: %{public}s", hex_dump);
        
        return false;
    }
    
    // 6. 验证上下文指针
    char ctx_str[32];
    snprintf(ctx_str, sizeof(ctx_str), "%p", &modelInfo.ctx);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, 
                 "上下文指针地址: %{public}s", ctx_str);
    
    // 7. 验证数据完整性（可选）
    if (modelData.size > 100) {
        uint32_t checksum = 0;
        for (size_t i = 0; i < 100; i++) {
            checksum += data_ptr[i];
        }
        char checksum_str[16];
        snprintf(checksum_str, sizeof(checksum_str), "0x%08X", checksum);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, 
                     "前100字节校验和: %{public}s", checksum_str);
    }
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, 
                 "参数验证完成");
    return valid_magic;
}

/**
 * 初始化RKNN模型
 * @param modelData 模型数据结构体
 * @param modelInfo 模型信息结构体引用，用于存储初始化后的模型信息
 * @return 初始化成功返回true，失败返回false
 */
static bool InitializeRknnModel(const RknnModelData& modelData, RknnModelInfo& modelInfo) {
      // 步骤1: 验证参数
//    if (!ValidateRknnInitParams(modelData, modelInfo)) {
//        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, 
//                     "初始化参数验证失败");
//        return false;
//    }
//    
//    // 步骤2: 打印调用参数
//    OH_LOG_Print(LOG_APP, LOG_DEBUG, GLOBAL_RESMGR, TAG, 
//                 "调用rknn_init参数:");
//    OH_LOG_Print(LOG_APP, LOG_DEBUG, GLOBAL_RESMGR, TAG, 
//                 "  ctx地址: %p", &modelInfo.ctx);
//    OH_LOG_Print(LOG_APP, LOG_DEBUG, GLOBAL_RESMGR, TAG, 
//                 "  data地址: %p", modelData.data.get());
//    OH_LOG_Print(LOG_APP, LOG_DEBUG, GLOBAL_RESMGR, TAG, 
//                 "  size: %zu", modelData.size);
//    OH_LOG_Print(LOG_APP, LOG_DEBUG, GLOBAL_RESMGR, TAG, 
//                 "  flags: 0");
//    OH_LOG_Print(LOG_APP, LOG_DEBUG, GLOBAL_RESMGR, TAG, 
//                 "  reserved: NULL");
    // 初始化RKNN模型
    int ret = rknn_init(&modelInfo.ctx, modelData.data.get(), modelData.size, 0, NULL);
    if (ret < 0) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG,"RKNN模型初始化失败, 错误码: %{public}d",ret);
        return false;
    }
    
    // 设置使用所有三个NPU核心
    ret = rknn_set_core_mask(modelInfo.ctx, RKNN_NPU_CORE_0_1_2);
    if (ret < 0) {
        OH_LOG_Print(LOG_APP, LOG_WARN, GLOBAL_RESMGR, TAG, "设置NPU核心失败，将使用默认配置");
    } else {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "成功设置使用所有三个NPU核心");
    }
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "RKNN模型初始化成功");
    
    // 查询模型输入输出信息
    ret = rknn_query(modelInfo.ctx, RKNN_QUERY_IN_OUT_NUM, &modelInfo.io_num, sizeof(modelInfo.io_num));
    if (ret != RKNN_SUCC) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to query IO numbers");
        rknn_destroy(modelInfo.ctx);
        return false;
    }
    
    // 分配并查询输入tensor属性
    modelInfo.input_attr = new rknn_tensor_attr[modelInfo.io_num.n_input];
    memset(modelInfo.input_attr, 0, sizeof(rknn_tensor_attr) * modelInfo.io_num.n_input);
    for (int i = 0; i < modelInfo.io_num.n_input; i++) {
        modelInfo.input_attr[i].index = i;
        ret = rknn_query(modelInfo.ctx, RKNN_QUERY_INPUT_ATTR, &(modelInfo.input_attr[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to query input attr %{public}d", i);
        }
    }
    
    // 分配并查询输出tensor属性
    modelInfo.output_attr = new rknn_tensor_attr[modelInfo.io_num.n_output];
    memset(modelInfo.output_attr, 0, sizeof(rknn_tensor_attr) * modelInfo.io_num.n_output);
    for (int i = 0; i < modelInfo.io_num.n_output; i++) {
        modelInfo.output_attr[i].index = i;
        ret = rknn_query(modelInfo.ctx, RKNN_QUERY_OUTPUT_ATTR, &(modelInfo.output_attr[i]), sizeof(rknn_tensor_attr));
        
        if (ret != RKNN_SUCC) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to query output attr %{public}d", i);
        }
    }
    
    // 获取模型输入尺寸
    modelInfo.channel = 3;
    if (modelInfo.input_attr[0].fmt == RKNN_TENSOR_NCHW) {
        modelInfo.channel = modelInfo.input_attr[0].dims[1];
        modelInfo.height = modelInfo.input_attr[0].dims[2];
        modelInfo.width = modelInfo.input_attr[0].dims[3];
    } else {
        modelInfo.height = modelInfo.input_attr[0].dims[1];
        modelInfo.width = modelInfo.input_attr[0].dims[2];
        modelInfo.channel = modelInfo.input_attr[0].dims[3];
    }
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Model input: height=%{public}d, width=%{public}d, channel=%{public}d", modelInfo.height, modelInfo.width, modelInfo.channel);
    
    return true;
}

/**
 * 执行RKNN推理
 * @param modelInfo 模型信息结构体
 * @param resized_img 预处理后的输入图像
 * @param detect_result_group 检测结果组引用，用于存储推理结果
 * @param pads 填充信息
 * @param min_scale 最小缩放比例
 * @param inference_time 推理时间引用，用于返回推理耗时
 * @return 推理成功返回true，失败返回false
 */
static bool RunRknnInference(const RknnModelInfo& modelInfo, const cv::Mat& resized_img, detect_result_group_t& detect_result_group, const BOX_RECT& pads, float min_scale, float& inference_time) {
    // 准备输入数据
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = modelInfo.width * modelInfo.height * modelInfo.channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = resized_img.data;
    
    // 设置输入
    int ret = rknn_inputs_set(modelInfo.ctx, modelInfo.io_num.n_input, inputs);
    if (ret != RKNN_SUCC) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "rknn_inputs_set failed! ret=%{public}d", ret);
        return false;
    }
    
    // 准备输出
    rknn_output outputs[modelInfo.io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < modelInfo.io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 0;
    }
    
    // 执行推理
    struct timeval start_time, stop_time;
    gettimeofday(&start_time, NULL);
    
    ret = rknn_run(modelInfo.ctx, NULL);
    if (ret != RKNN_SUCC) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "rknn_run failed! ret=%{public}d", ret);
        return false;
    }
    
    ret = rknn_outputs_get(modelInfo.ctx, modelInfo.io_num.n_output, outputs, NULL);
    if (ret != RKNN_SUCC) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "rknn_outputs_get failed! ret=%{public}d", ret);
        return false;
    }
    
    gettimeofday(&stop_time, NULL);
    inference_time = (__get_us(stop_time) - __get_us(start_time)) / 1000;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Inference time: %{public}f ms", inference_time);
    
    // 后处理
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i < modelInfo.io_num.n_output; ++i) {
        out_scales.push_back(modelInfo.output_attr[i].scale);
        out_zps.push_back(modelInfo.output_attr[i].zp);
    }
    
    const float nms_threshold = YOLOV5_NMS_THRESH;
    const float box_conf_threshold = YOLOV5_BOX_THRESH;
    
    if (modelInfo.io_num.n_output >= 3) {
        post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, 
                    modelInfo.height, modelInfo.width, box_conf_threshold, nms_threshold, pads, min_scale, min_scale, 
                    out_zps, out_scales, &detect_result_group);
        
        // 记录检测结果，但不在图片上画框
        for (int i = 0; i < detect_result_group.count; i++) {
            detect_result_t *det_result = &(detect_result_group.results[i]);
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Detected: %{public}s %{public}.1f%% at [%{public}d,%{public}d,%{public}d,%{public}d]", 
                       det_result->name, det_result->prop * 100, 
                       det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom);
        }
    }
    
    // 释放输出资源
    rknn_outputs_release(modelInfo.ctx, modelInfo.io_num.n_output, outputs);
    
    return true;
}

/**
 * 创建错误返回结果的辅助函数
 * @param env NAPI环境
 * @param result 结果对象
 * @param errorMessage 错误消息
 */
static void CreateErrorResult(napi_env env, napi_value result, const char* errorMessage) {
    napi_value success, message;
    napi_get_boolean(env, false, &success);
    napi_create_string_utf8(env, errorMessage, NAPI_AUTO_LENGTH, &message);
    napi_set_named_property(env, result, "success", success);
    napi_set_named_property(env, result, "message", message);
}

/**
 * 图片处理函数：读取、预处理、缩放和填充
 * @param imagePath 图像文件路径
 * @param modelInfo 模型信息结构体
 * @param orig_img 原始图像引用，用于存储读取的图像
 * @param resized_img 调整后图像引用，用于存储预处理后的图像
 * @param pads 填充信息引用
 * @param min_scale 最小缩放比例引用
 * @param img_width 图像宽度引用
 * @param img_height 图像高度引用
 * @return 处理成功返回true，失败返回false
 */
static bool yolov5_img_processing(const std::string& imagePath, const RknnModelInfo& modelInfo, 
                                  cv::Mat& orig_img, cv::Mat& resized_img, BOX_RECT& pads, float& min_scale, 
                                  int& img_width, int& img_height) {
    // 读取图片
    orig_img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (!orig_img.data) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to read %{public}s", imagePath.c_str());
        return false;
    }
    
    // 图片预处理
    cv::Mat img_rgb;
    cv::cvtColor(orig_img, img_rgb, cv::COLOR_BGR2RGB);
    img_width = img_rgb.cols;
    img_height = img_rgb.rows;
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Original image: width=%{public}d, height=%{public}d", img_width, img_height);
    
    // 图片缩放和填充
    cv::Size target_size(modelInfo.width, modelInfo.height);
    resized_img = cv::Mat(target_size.height, target_size.width, CV_8UC3);
    float scale_w = (float)target_size.width / img_width;
    float scale_h = (float)target_size.height / img_height;
    min_scale = std::min(scale_w, scale_h);
    
    letterbox(img_rgb, resized_img, pads, min_scale, target_size);
    
    return true;
}

/**
 * 清理RKNN模型资源
 * @param modelInfo 模型信息结构体引用
 */
static void CleanupRknnModel(RknnModelInfo& modelInfo) {
    if (modelInfo.input_attr) {
        delete[] modelInfo.input_attr;
        modelInfo.input_attr = nullptr;
    }
    if (modelInfo.output_attr) {
        delete[] modelInfo.output_attr;
        modelInfo.output_attr = nullptr;
    }
    rknn_destroy(modelInfo.ctx);
}

/**
 * 主要的图像处理接口，执行人脸检测和识别
 * @param env NAPI环境
 * @param info 回调信息
 * @return NAPI值，包含处理结果
 */
static napi_value ProcessImageWithRKNN(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "ProcessImageWithRKNN Begin");
    
    // 获取参数信息
    size_t argc = 3; // photoUri, resourceManager, sandboxDir
    napi_value argv[3] = { nullptr };
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    
    // 检查参数数量
    if (argc < 3) {
        napi_throw_error(env, nullptr, "需要提供photoUri、resourceManager和sandboxDir参数");
        return nullptr;
    }
    
    // 创建返回对象
    napi_value result;
    napi_create_object(env, &result);
    
    try {
         OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "-----snw----111---");
        auto proc_yolo_start = std::chrono::steady_clock::now();
             OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "-----snw----2---");
        // 获取图片路径
        char photoUri[512];
        size_t photoUriLength;
        napi_get_value_string_utf8(env, argv[0], photoUri, sizeof(photoUri), &photoUriLength);
        
        // 移除file://前缀
        std::string imagePath(photoUri);
        if (imagePath.find("file://") == 0) {
            imagePath = imagePath.substr(7);
        }
        
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Processing image: %{public}s", imagePath.c_str());
        
        // 获取ResourceManager
        NativeResourceManager *mNativeResMgr = OH_ResourceManager_InitNativeResourceManager(env, argv[1]);
        if (mNativeResMgr == nullptr) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to initialize NativeResourceManager");
            CreateErrorResult(env, result, "ResourceManager初始化失败");
            return result;
        }
        
        // 获取沙箱目录路径
        char sandboxDir[256];
        size_t sandboxDirLength;
        napi_get_value_string_utf8(env, argv[2], sandboxDir, sizeof(sandboxDir), &sandboxDirLength);
        
        // 加载RKNN模型数据
//        RknnModelData modelData;
//        if (!LoadRknnModelData(std::string(sandboxDir), mNativeResMgr, modelData)) {
//            OH_ResourceManager_ReleaseNativeResourceManager(mNativeResMgr);
//            CreateErrorResult(env, result, "模型数据加载失败");
//            return result;
//        }
        if(load_data_flag == false){
            load_data_flag = LoadRknnModelData(std::string(sandboxDir), mNativeResMgr, g_yolo_model_data);
            if(load_data_flag == false){
                OH_ResourceManager_ReleaseNativeResourceManager(mNativeResMgr);
                CreateErrorResult(env, result, "模型数据加载失败");
                return result;
            }
        }
        
        
        // 释放ResourceManager
        OH_ResourceManager_ReleaseNativeResourceManager(mNativeResMgr);
        
        // 初始化RKNN模型
//        RknnModelInfo modelInfo;
//        if (!InitializeRknnModel(g_yolo_model_data, modelInfo)) {
//            CreateErrorResult(env, result, "RKNN模型初始化失败");
//            return result;
//        }
        if(Rknn_init_flag == false){
            Rknn_init_flag = InitializeRknnModel(g_yolo_model_data, g_yolo_model_info);
            if(Rknn_init_flag == false){
                CreateErrorResult(env, result, "RKNN模型初始化失败");
                return result;
            }
        }
        // 图片处理：读取、预处理、缩放和填充
        cv::Mat orig_img, resized_img;
        BOX_RECT pads;
        float min_scale;
        int img_width, img_height;
        
        if (!yolov5_img_processing(imagePath, g_yolo_model_info, orig_img, resized_img, pads, min_scale, img_width, img_height)) {
//            CleanupRknnModel(modelInfo);
            CreateErrorResult(env, result, "图片处理失败");
            return result;
        }
        
        // 执行RKNN推理
        detect_result_group_t detect_result_group;
        float inference_time;
        if (!RunRknnInference(g_yolo_model_info, resized_img, detect_result_group, pads, min_scale, inference_time)) {
//            CleanupRknnModel(modelInfo);
            CreateErrorResult(env, result, "RKNN推理执行失败");
            return result;
        }
        
        // 生成输出文件路径（使用时间戳命名）
        struct timeval tv;
        gettimeofday(&tv, NULL);
        long timestamp = tv.tv_sec * 1000 + tv.tv_usec / 1000;
        std::string out_path = std::string(sandboxDir) + "/out_" + std::to_string(timestamp) + ".jpg";
        
        // 保存结果图片
        bool saveResult = cv::imwrite(out_path, orig_img);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "-----snw----3---");
        auto total_yolo_end = std::chrono::steady_clock::now();
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "-----snw----4---");
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_yolo_end - proc_yolo_start).count();
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, 
                     "----yolo---- Total processing time: %{public}lld ms", total_ms);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "-----snw----5---");
        if (!saveResult) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to save result image");
        } else {
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Result saved to: %{public}s", out_path.c_str());
            
            // 推理完成并生成输出图片后，删除预览帧图片（除了当前正在处理的文件）
            try {
                // 从输入路径中提取当前正在处理的文件名
                std::string currentFileName = "";
                size_t lastSlash = imagePath.find_last_of("/\\");
                if (lastSlash != std::string::npos) {
                    currentFileName = imagePath.substr(lastSlash + 1);
                }
                
                // 获取沙箱目录中的文件列表
                DIR* dir = opendir(sandboxDir);
                if (dir != nullptr) {
                    struct dirent* entry;
                    std::vector<std::string> out_files;
                    
                    while ((entry = readdir(dir)) != nullptr) {
                        std::string filename = entry->d_name;
                        // 检查是否为预览帧图片
                        if (filename.find("preview_frame_") == 0 && filename.find(".jpg") != std::string::npos) {
                            // 跳过当前正在处理的文件
                            if (filename == currentFileName) {
                                OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Skipping current processing file: %{public}s", filename.c_str());
                                continue;
                            }
                            
                            std::string file_path = std::string(sandboxDir) + "/" + filename;
                            // 删除预览帧图片
                            if (remove(file_path.c_str()) == 0) {
                                OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Deleted preview frame: %{public}s", file_path.c_str());
                            } else {
                                OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to delete preview frame: %{public}s", file_path.c_str());
                            }
                            
                        }
                        // 收集out*.jpg文件
                        else if (filename.find("out_") == 0 && filename.find(".jpg") != std::string::npos) {
                            out_files.push_back(filename);
                        }
                    }
                    closedir(dir);
                    
                    // 检查out*.jpg文件数量，如果超过20张则删除最早的10张
                    if (out_files.size() > 20) {
                        // 按文件名排序（文件名包含时间戳，所以按字典序排序即可得到时间顺序）
                        std::sort(out_files.begin(), out_files.end());
                        
                        // 删除最早的10张图片
                        for (int i = 0; i < 10 && i < out_files.size(); i++) {
                            std::string file_path = std::string(sandboxDir) + "/" + out_files[i];
                            if (remove(file_path.c_str()) == 0) {
                                OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Deleted old output file: %{public}s", file_path.c_str());
                            } else {
                                OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to delete old output file: %{public}s", file_path.c_str());
                            }
                        }
                        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Cleaned up old output files, total files before: %{public}d", (int)out_files.size());
                    }
                }
            } catch (const std::exception& e) {
                OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Exception while deleting preview frames: %{public}s", e.what());
            }
        }
        
        // 释放RKNN模型资源
//        CleanupRknnModel(g_yolo_model_info);
        
        // 构建返回结果
        napi_value success, message, savedImagePath, imageInfo, inferenceTimeValue, detections;
        napi_get_boolean(env, true, &success);
        napi_create_string_utf8(env, "RKNN图像处理完成", NAPI_AUTO_LENGTH, &message);
        napi_create_string_utf8(env, out_path.c_str(), NAPI_AUTO_LENGTH, &savedImagePath);
        napi_create_double(env, inference_time, &inferenceTimeValue);
        
        // 图像信息
        napi_create_object(env, &imageInfo);
        napi_value width_value, height_value, channel_value;
        napi_create_int32(env, img_width, &width_value);
        napi_create_int32(env, img_height, &height_value);
        napi_create_int32(env, 3, &channel_value);
        napi_set_named_property(env, imageInfo, "width", width_value);
        napi_set_named_property(env, imageInfo, "height", height_value);
        napi_set_named_property(env, imageInfo, "channel", channel_value);
        
        
        // 使用FaceCropUtils进行人脸裁切
        int face_count = FaceCropUtils::cropAndSaveFaces(orig_img, detect_result_group, std::string(sandboxDir), imagePath);
        
        // 启动异步人脸识别工作线程（如果尚未启动）
        start_face_recognition_worker();
        
        // 对刚裁切生成的人脸进行异步识别对比（只处理刚生成的图片）
        if (face_count > 0) {
            try {
                DIR* dir = opendir(sandboxDir);
                if (dir != nullptr) {
                    struct dirent* entry;
                    std::vector<std::pair<std::string, time_t>> face_files_with_time;
                    
                    // 获取当前时间戳，用于筛选刚生成的文件
                    time_t current_time = time(nullptr);
                    
                    // 查找face*_preprocessed.jpg文件
                    while ((entry = readdir(dir)) != nullptr) {
                        std::string filename = entry->d_name;
                        // 查找face开头且包含_preprocessed.jpg的文件
                        if (filename.find("face") == 0 && filename.find("_preprocessed.jpg") != std::string::npos) {
                            
                            // 获取文件的修改时间
                            std::string full_path = std::string(sandboxDir) + "/" + filename;
                            struct stat file_stat;
                            if (stat(full_path.c_str(), &file_stat) == 0) {
                                // 只处理最近1秒内创建/修改的文件（刚裁切生成的）
                                if (current_time - file_stat.st_mtime <= 1) {
                                    face_files_with_time.push_back({filename, file_stat.st_mtime});
                                }
                            }
                        }
                    }
                    closedir(dir);
                    
                    // 按修改时间排序，优先处理最新的文件
                    std::sort(face_files_with_time.begin(), face_files_with_time.end(), 
                             [](const std::pair<std::string, time_t>& a, const std::pair<std::string, time_t>& b) {
                                 return a.second > b.second;
                             });
                    
                    // 将刚生成的人脸文件添加到异步处理队列，限制为face_count数量
                    for (size_t i = 0; i < face_files_with_time.size() && i < static_cast<size_t>(face_count); i++) {
                        const std::string& face_filename = face_files_with_time[i].first;
                        std::string face_path = std::string(sandboxDir) + "/" + face_filename;
                        add_face_recognition_task(face_path, std::string(sandboxDir), mNativeResMgr);
                        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Added face recognition task for newly processed file: %{public}s", face_filename.c_str());
                    }
                }
            } catch (const std::exception& e) {
                OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Exception while adding face recognition tasks: %{public}s", e.what());
            }
        }
        
        // 获取最新的异步识别结果（如果有的话）
        float similarity;
        bool is_same_person, has_result;
        std::string processed_file, matched_reference_path;
        
        if (get_latest_face_recognition_result(similarity, is_same_person, has_result, processed_file, matched_reference_path)) {
            // 添加相似度到返回结果
            napi_value similarity_value;
            napi_create_double(env, similarity, &similarity_value);
            napi_set_named_property(env, result, "faceSimilarity", similarity_value);
            
            // 添加是否为同一人的判断
            napi_value is_same_person_value;
            napi_get_boolean(env, is_same_person, &is_same_person_value);
            napi_set_named_property(env, result, "isSamePerson", is_same_person_value);
            
            // 添加匹配的参考人脸路径
            if (!matched_reference_path.empty()) {
                napi_value matched_reference_path_value;
                napi_create_string_utf8(env, matched_reference_path.c_str(), NAPI_AUTO_LENGTH, &matched_reference_path_value);
                napi_set_named_property(env, result, "matchedReferencePath", matched_reference_path_value);
            }
            
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Using latest face recognition result: %{public}s (similarity: %{public}f)", 
                       is_same_person ? "Same person" : "Different person", similarity);
        }
        
        // 检测结果数组
        napi_create_array(env, &detections);
        int detection_index = 0;
        
        // 创建人脸图片路径数组
        napi_value faceImagePaths;
        napi_create_array(env, &faceImagePaths);
        
        for (int i = 0; i < detect_result_group.count; i++) {
            detect_result_t *det_result = &(detect_result_group.results[i]);
            // 只返回face类别的检测结果
            if (strcmp(det_result->name, "face") == 0) {
                napi_value detection;
                napi_create_object(env, &detection);
                
                napi_value name, confidence, left_val, top_val, right_val, bottom_val;
                napi_create_string_utf8(env, det_result->name, NAPI_AUTO_LENGTH, &name);
                napi_create_double(env, det_result->prop, &confidence);
                napi_create_int32(env, det_result->box.left, &left_val);
                napi_create_int32(env, det_result->box.top, &top_val);
                napi_create_int32(env, det_result->box.right, &right_val);
                napi_create_int32(env, det_result->box.bottom, &bottom_val);
                
                napi_set_named_property(env, detection, "name", name);
                napi_set_named_property(env, detection, "confidence", confidence);
                napi_set_named_property(env, detection, "left", left_val);
                napi_set_named_property(env, detection, "top", top_val);
                napi_set_named_property(env, detection, "right", right_val);
                napi_set_named_property(env, detection, "bottom", bottom_val);
                
                napi_set_element(env, detections, detection_index, detection);
                detection_index++;
            }
        }
        
        // 添加人脸图片路径到返回结果 - 动态读取带时间戳的文件
        if (face_count > 0) {
            DIR* face_dir = opendir(sandboxDir);
            if (face_dir != nullptr) {
                struct dirent* entry;
                std::vector<std::string> face_files;
                
                // 收集所有人脸文件
                while ((entry = readdir(face_dir)) != nullptr) {
                    std::string filename = entry->d_name;
                    if (filename.find("face") == 0 && filename.find(".jpg") != std::string::npos) {
                        face_files.push_back(std::string(sandboxDir) + "/" + filename);
                    }
                }
                closedir(face_dir);
                
                // 按文件名排序以确保顺序一致
                std::sort(face_files.begin(), face_files.end());
                
                // 添加到返回结果
                for (size_t i = 0; i < face_files.size() && i < static_cast<size_t>(face_count); i++) {
                    napi_value face_path_value;
                    napi_create_string_utf8(env, face_files[i].c_str(), NAPI_AUTO_LENGTH, &face_path_value);
                    napi_set_element(env, faceImagePaths, i, face_path_value);
                }
            }
        }
        
        if (face_count > 0) {
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Successfully processed and saved %{public}d face(s) using FaceCropUtils", face_count);
        }
        
        napi_set_named_property(env, result, "success", success);
        napi_set_named_property(env, result, "message", message);
        napi_set_named_property(env, result, "savedImagePath", savedImagePath);
        napi_set_named_property(env, result, "imageInfo", imageInfo);
        napi_set_named_property(env, result, "inferenceTime", inferenceTimeValue);
        napi_set_named_property(env, result, "detections", detections);
        napi_set_named_property(env, result, "faceImagePaths", faceImagePaths);
        
    } catch (const std::exception& e) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Exception in ProcessImageWithRKNN: %{public}s", e.what());
        CreateErrorResult(env, result, "处理过程中发生异常");
    }
    
    return result;
}

/**
 * 保存模型文件到沙箱目录
 * @param env NAPI环境
 * @param info 回调信息
 * @return NAPI值，包含保存结果
 */
static napi_value SaveModelFile(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "SaveModelFile Begin");
    
    // 创建返回对象
    napi_value result;
    napi_create_object(env, &result);
    
    // 获取参数信息
    size_t argc = 4; // resourceManager, sandboxDir, rawfilePath, savePath
    napi_value argv[4] = { nullptr };
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    
    // 检查参数数量
    if (argc < 2) {
        CreateErrorResult(env, result, "需要提供ResourceManager和沙箱目录路径参数");
        return result;
    }
    
    // 获取ResourceManager
    NativeResourceManager *mNativeResMgr = OH_ResourceManager_InitNativeResourceManager(env, argv[0]);
    if (mNativeResMgr == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to initialize NativeResourceManager");
        CreateErrorResult(env, result, "ResourceManager初始化失败");
        return result;
    }
    
    // 获取沙箱目录路径
    char sandboxDir[256];
    size_t sandboxDirLength;
    napi_get_value_string_utf8(env, argv[1], sandboxDir, sizeof(sandboxDir), &sandboxDirLength);
    
    std::string rawfile_path, save_path;
    
    // 如果提供了4个参数，使用自定义路径
    if (argc >= 4) {
        char rawfilePath[512], savePath[512];
        size_t rawfilePathLength, savePathLength;
        napi_get_value_string_utf8(env, argv[2], rawfilePath, sizeof(rawfilePath), &rawfilePathLength);
        napi_get_value_string_utf8(env, argv[3], savePath, sizeof(savePath), &savePathLength);
        
        rawfile_path = std::string(rawfilePath);
        save_path = std::string(savePath);
    } else {
        // 默认保存yolov5和facenet模型
        rawfile_path = "rawfile/model/usemodel/yolov5.rknn";
        save_path = std::string(sandboxDir) + "/yolov5_model.rknn";
    }
    
    // 使用通用函数加载和保存模型
    bool success = load_and_save_model(mNativeResMgr, rawfile_path, save_path);
    
    if (!success) {
        OH_ResourceManager_ReleaseNativeResourceManager(mNativeResMgr);
        CreateErrorResult(env, result, "模型文件保存失败");
        return result;
    }
    
    // 如果是默认模式，同时保存facenet模型
    if (argc < 4) {
        std::string facenet_rawfile_path = "rawfile/model/usemodel/facenet.rknn";
        std::string facenet_save_path = std::string(sandboxDir) + "/facenet.rknn";
        
        bool facenet_success = load_and_save_model(mNativeResMgr, facenet_rawfile_path, facenet_save_path);
        if (!facenet_success) {
            OH_LOG_Print(LOG_APP, LOG_WARN, GLOBAL_RESMGR, TAG, "Facenet model save failed, but continuing...");
        }
    }
    
    OH_ResourceManager_ReleaseNativeResourceManager(mNativeResMgr);
    
    // 返回成功结果
    napi_value napi_success, message, savedPath;
    napi_get_boolean(env, true, &napi_success);
    napi_create_string_utf8(env, "模型文件保存成功", NAPI_AUTO_LENGTH, &message);
    napi_create_string_utf8(env, save_path.c_str(), NAPI_AUTO_LENGTH, &savedPath);
    
    napi_set_named_property(env, result, "success", napi_success);
    napi_set_named_property(env, result, "message", message);
    napi_set_named_property(env, result, "savedModelPath", savedPath);
    
    return result;
}

/**
 * 保存人脸数据库图片到沙箱目录
 * @param env NAPI环境
 * @param info 回调信息
 * @return NAPI值，包含保存结果
 */
static napi_value SaveFaceDatabaseImages(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "SaveFaceDatabaseImages Begin");
    
    // 创建返回对象
    napi_value result;
    napi_create_object(env, &result);
    
    // 获取参数信息
    size_t argc = 2;
    napi_value argv[2] = { nullptr };
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    
    // 检查参数数量
    if (argc < 2) {
        CreateErrorResult(env, result, "需要提供ResourceManager和沙箱目录路径参数");
        return result;
    }
    
    // 获取ResourceManager
    NativeResourceManager *mNativeResMgr = OH_ResourceManager_InitNativeResourceManager(env, argv[0]);
    if (mNativeResMgr == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to initialize NativeResourceManager");
        CreateErrorResult(env, result, "ResourceManager初始化失败");
        return result;
    }
    
    // 获取沙箱目录路径
    char sandboxDir[256];
    size_t sandboxDirLength;
    napi_get_value_string_utf8(env, argv[1], sandboxDir, sizeof(sandboxDir), &sandboxDirLength);
    
    int savedCount = 0;
    int totalAttempted = 0;
    
    // 尝试保存人脸数据库图片（1.jpg到20.jpg）
    for (int i = 1; i <= 20; i++) {
        std::string rawfilePath = "rawfile/face_databases/" + std::to_string(i) + ".jpg";
        std::string savePath = std::string(sandboxDir) + "/" + std::to_string(i) + ".jpg";
        
        totalAttempted++;
        
        // 尝试打开rawfile中的图片
        RawFile *imageRawFile = OH_ResourceManager_OpenRawFile(mNativeResMgr, rawfilePath.c_str());
        if (imageRawFile == nullptr) {
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Image file not found: %{public}s", rawfilePath.c_str());
            continue;
        }
        
        // 获取图片数据大小并读取
        long imageSize = OH_ResourceManager_GetRawFileSize(imageRawFile);
        std::unique_ptr<uint8_t[]> imageData = std::make_unique<uint8_t[]>(imageSize);
        int readResult = OH_ResourceManager_ReadRawFile(imageRawFile, imageData.get(), imageSize);
        OH_ResourceManager_CloseRawFile(imageRawFile);
        
        if (readResult != imageSize) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to read image data: %{public}s", rawfilePath.c_str());
            continue;
        }
        
        // 将图片数据保存到沙箱目录
        FILE *imageFile = fopen(savePath.c_str(), "wb");
        if (imageFile == nullptr) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to create image file: %{public}s", savePath.c_str());
            continue;
        }
        
        size_t written = fwrite(imageData.get(), 1, imageSize, imageFile);
        fclose(imageFile);
        
        if (written != imageSize) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to write image data to file: %{public}s", savePath.c_str());
            continue;
        }
        
        savedCount++;
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Image saved successfully: %{public}s -> %{public}s", rawfilePath.c_str(), savePath.c_str());
    }
    
    // 如果有图片保存成功，则对沙箱中的数字命名图片进行预处理
    int preprocessedCount = 0;
    if (savedCount > 0) {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Starting preprocessing of saved face database images");
        preprocessedCount = FaceCropUtils::preprocessNumberedImages(std::string(sandboxDir), std::string(sandboxDir));
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Preprocessing completed. %{public}d images processed", preprocessedCount);
        
        // 检查是否存在{数字}_preprocessed.jpg文件，如果存在且未处理过则进行Facenet推理（只在应用启动时执行一次）
        if (preprocessedCount > 0 && !get_numbered_images_processed()) {
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Starting Facenet inference for numbered preprocessed images");
            int feature_count = 0;
            
            for (int i = 1; i <= 20; ++i) { // 检查1到20的数字命名图片
                std::string preprocessed_filename = std::to_string(i) + "_preprocessed.jpg";
                std::string preprocessed_path = std::string(sandboxDir) + "/" + preprocessed_filename;
                
                struct stat file_stat;
                if (stat(preprocessed_path.c_str(), &file_stat) == 0) {
                    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Found %{public}s, performing Facenet inference", preprocessed_filename.c_str());
                    
                    // 读取预处理后的人脸图像
                    cv::Mat face_image = cv::imread(preprocessed_path, cv::IMREAD_COLOR);
                    if (!face_image.data) {
                        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Failed to read %{public}s", preprocessed_filename.c_str());
                        continue;
                    }
                    
                    std::vector<float> current_feature;
                    int facenet_ret = facenet_inference(face_image, current_feature, mNativeResMgr);
                    
                    if (facenet_ret == 0) {
                        // 将特征向量分别添加到参考特征中
                        add_reference_face_feature(current_feature, preprocessed_path);
                        feature_count++;
                        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Feature vector from %{public}s has been added to reference features", preprocessed_filename.c_str());
                    } else {
                        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Facenet inference failed for %{public}s", preprocessed_filename.c_str());
                    }
                }
            }
            
            if (feature_count > 0) {
                OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Successfully extracted %{public}d feature vectors from numbered preprocessed images (total reference features: %{public}zu)", feature_count, get_reference_face_count());
                // 自动切换到识别模式
                set_face_recognition_mode(false);
                OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Switched to recognition mode after processing numbered images");
            }
            
            // 标记已处理，确保只在应用启动时执行一次
            set_numbered_images_processed(true);
        }
    }
    
    // 在所有使用mNativeResMgr的操作完成后释放ResourceManager
    OH_ResourceManager_ReleaseNativeResourceManager(mNativeResMgr);
    
    // 返回结果
    napi_value success, message, savedCountValue, totalAttemptedValue;
    napi_get_boolean(env, savedCount > 0, &success);
    
    std::string messageStr = "成功保存 " + std::to_string(savedCount) + " 张图片，共尝试 " + std::to_string(totalAttempted) + " 张";
    if (preprocessedCount > 0) {
        messageStr += "，预处理 " + std::to_string(preprocessedCount) + " 张图片";
        if (get_numbered_images_processed() && get_reference_face_count() > 0) {
            messageStr += "，提取 " + std::to_string(get_reference_face_count()) + " 个特征向量";
        }
    }
    napi_create_string_utf8(env, messageStr.c_str(), NAPI_AUTO_LENGTH, &message);
    napi_create_int32(env, savedCount, &savedCountValue);
    napi_create_int32(env, totalAttempted, &totalAttemptedValue);
    
    napi_set_named_property(env, result, "success", success);
    napi_set_named_property(env, result, "message", message);
    napi_set_named_property(env, result, "savedCount", savedCountValue);
    napi_set_named_property(env, result, "totalAttempted", totalAttemptedValue);
    
    return result;
}

/**
 * 初始化NAPI模块，定义导出的函数
 * @param env NAPI环境
 * @param exports 导出对象
 * @return 导出对象
 */
napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        // getRawFileContent已删除
        {"processImageWithRKNN", nullptr, ProcessImageWithRKNN, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"saveModelFile", nullptr, SaveModelFile, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"saveFaceDatabaseImages", nullptr, SaveFaceDatabaseImages, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}

static napi_module demoModule = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "entry",
    .nm_priv = ((void*)0),
    .reserved = { 0 },
};

/**
 * 模块注册函数（构造函数），在模块加载时自动调用
 */
extern "C" __attribute__((constructor)) void RegisterEntryModule(void)
{
    napi_module_register(&demoModule);
    
    // 应用启动时清空所有face*_*_preprocessed.jpg和face*_*.jpg文件
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Application startup: cleaning up face files");
    
    // 获取沙盒目录路径
    std::string sandbox_dir = "/data/storage/el2/base/haps/entry/files";
    cleanup_all_face_preprocessed_files(sandbox_dir);
}

/**
 * 模块注销函数（析构函数），在模块卸载时自动调用
 */
extern "C" __attribute__((destructor)) void UnregisterEntryModule(void)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Module cleanup: stopping face recognition worker thread");
    stop_face_recognition_worker();
}
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

#ifndef MODEL_H
#define MODEL_H

#include <rawfile/raw_file_manager.h>
#include <string>
#include <vector>
#include "opencv2/core/core.hpp"

/**
 * 模型工具类
 * 提供模型文件加载和保存功能
 */

/**
 * 加载并保存模型文件
 * @param mNativeResMgr 资源管理器指针
 * @param rawfile_path 原始文件路径
 * @param save_path 保存路径
 * @return 成功返回true，失败返回false
 */
bool load_and_save_model(NativeResourceManager* mNativeResMgr, const std::string& rawfile_path, const std::string& save_path);

/**
 * Facenet推理函数
 * @param face_image 输入的人脸图像
 * @param feature_vector 输出的特征向量
 * @param mNativeResMgr 资源管理器
 * @return 0表示成功，-1表示失败
 */
int facenet_inference(const cv::Mat& face_image, std::vector<float>& feature_vector, NativeResourceManager* mNativeResMgr);

/**
 * YOLOv5推理函数
 * @param input_image 输入图像
 * @param detection_results 检测结果
 * @param mNativeResMgr 资源管理器
 * @param sandbox_dir 沙盒目录路径
 * @return 0表示成功，-1表示失败
 */
int yolov5_inference(const cv::Mat& input_image, std::vector<std::string>& detection_results, NativeResourceManager* mNativeResMgr, const std::string& sandbox_dir);

/**
 * 启动异步人脸识别工作线程
 */
void start_face_recognition_worker();

/**
 * 停止异步人脸识别工作线程
 */
void stop_face_recognition_worker();

/**
 * 添加人脸识别任务到队列
 * @param face_path 人脸图像路径
 * @param sandbox_dir 沙盒目录
 * @param resource_manager 资源管理器
 */
void add_face_recognition_task(const std::string& face_path, const std::string& sandbox_dir, NativeResourceManager* resource_manager);

/**
 * 获取最新的人脸识别结果
 * @param similarity 相似度
 * @param is_same_person 是否为同一人
 * @param has_result 是否有结果
 * @param processed_file 处理的文件路径
 * @param matched_reference_path 匹配的参考图片路径
 * @return true表示有新结果，false表示无新结果
 */
bool get_latest_face_recognition_result(float& similarity, bool& is_same_person, bool& has_result, std::string& processed_file, std::string& matched_reference_path);

/**
 * 添加参考人脸特征
 * @param feature_vector 特征向量
 * @param image_path 图片路径
 */
void add_reference_face_feature(const std::vector<float>& feature_vector, const std::string& image_path);

/**
 * 清空参考人脸特征
 */
void clear_reference_face_features();

/**
 * 获取参考人脸特征数量
 * @return 特征数量
 */
size_t get_reference_face_count();

/**
 * 设置人脸识别模式
 * @param generate_mode true为特征录入模式，false为识别模式
 */
void set_face_recognition_mode(bool generate_mode);

/**
 * 获取人脸识别模式
 * @return true为特征录入模式，false为识别模式
 */
bool get_face_recognition_mode();

/**
 * 设置数字命名图片处理状态
 * @param processed 是否已处理
 */
void set_numbered_images_processed(bool processed);

/**
 * 获取数字命名图片处理状态
 * @return 是否已处理
 */
bool get_numbered_images_processed();

#endif // MODEL_H
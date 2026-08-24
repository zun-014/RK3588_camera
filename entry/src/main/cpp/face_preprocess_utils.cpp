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
#include "face_preprocess_utils.h"
#include "hilog/log.h"
#include <dirent.h>
#include <algorithm>

#define FACE_PREPROCESS_TAG "FacePreprocessUtils"
const int GLOBAL_RESMGR = 0xFF00;

// 构造函数，计算缩放比例和填充大小
FacePreprocessUtils::FacePreprocessUtils(int width, int height, int target_size) {
    // 根据目标大小计算缩放比例
    scale_ = static_cast<double>(target_size) / std::max(height, width);

    // 根据缩放比例计算填充的大小
    padding_x_ = target_size - static_cast<int>(width * scale_);
    padding_y_ = target_size - static_cast<int>(height * scale_);

    // 计算新的尺寸
    new_size_ = cv::Size(static_cast<int>(width * scale_), static_cast<int>(height * scale_));

    target_size_ = target_size;

    // 设置填充信息
    letterbox_.scale = scale_;
    letterbox_.x_pad = padding_x_ / 2;
    letterbox_.y_pad = padding_y_ / 2;
}

/**
 * 将输入图像转换为目标大小，保持宽高比并进行填充
 * @param src 输入的源图像
 * @return 转换后的图像指针，失败时返回nullptr
 */
std::unique_ptr<cv::Mat> FacePreprocessUtils::convert(const cv::Mat& src) {
    if (src.empty()) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, FACE_PREPROCESS_TAG,
                   "Input image is empty");
        return nullptr;
    }

    // 调整图像大小
    cv::Mat resize_img;
    cv::resize(src, resize_img, new_size_);

    // 创建一个新的目标大小的空白图像，使用填充颜色(114, 114, 114)与ImageProcess.cpp保持一致
    auto square_img = std::make_unique<cv::Mat>(target_size_, target_size_, src.type(), cv::Scalar(114, 114, 114));

    // 计算填充位置
    cv::Point position(padding_x_ / 2, padding_y_ / 2);

    // 将调整大小后的图像复制到目标图像
    resize_img.copyTo((*square_img)(cv::Rect(position.x, position.y, resize_img.cols, resize_img.rows)));

    return square_img;
}

// 获取letterbox配置信息
const letterbox_t& FacePreprocessUtils::getLetterBox() const {
    return letterbox_;
}

/**
 * 批量预处理指定目录中的人脸图像文件
 * @param face_files_dir 输入目录路径，包含待处理的人脸图像
 * @param target_size 目标图像尺寸
 * @param output_dir 输出目录路径，用于保存预处理后的图像
 * @return 成功处理的图像数量
 */
int FacePreprocessUtils::preprocessFaceImages(const std::string& face_files_dir, 
                                             int target_size, 
                                             const std::string& output_dir) {
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, FACE_PREPROCESS_TAG, 
               "Starting batch preprocessing of face images in directory: %{public}s", 
               face_files_dir.c_str());
    
    DIR* dir = opendir(face_files_dir.c_str());
    if (dir == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, FACE_PREPROCESS_TAG, 
                   "Failed to open directory: %{public}s", face_files_dir.c_str());
        return 0;
    }
    
    struct dirent* entry;
    std::vector<std::string> face_files;
    
    // 收集所有人脸图像文件
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.find("face") == 0 && filename.find(".jpg") != std::string::npos) {
            face_files.push_back(filename);
        }
    }
    closedir(dir);
    
    // 按文件名排序
    std::sort(face_files.begin(), face_files.end());
    
    int processed_count = 0;
    
    for (const auto& filename : face_files) {
        std::string input_path = face_files_dir + "/" + filename;
        
        // 生成预处理后的文件名
        size_t dot_pos = filename.find_last_of(".");
        std::string base_name = filename.substr(0, dot_pos);
        std::string output_filename = base_name + "_preprocessed.jpg";
        std::string output_path = output_dir + "/" + output_filename;
        
        if (preprocessSingleFace(input_path, target_size, output_path)) {
            processed_count++;
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, FACE_PREPROCESS_TAG, 
                       "Successfully preprocessed: %{public}s -> %{public}s", 
                       filename.c_str(), output_filename.c_str());
        } else {
            OH_LOG_Print(LOG_APP, LOG_WARN, GLOBAL_RESMGR, FACE_PREPROCESS_TAG, 
                       "Failed to preprocess: %{public}s", filename.c_str());
        }
    }
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, FACE_PREPROCESS_TAG, 
               "Batch preprocessing completed. Processed %{public}d out of %{public}d files", 
               processed_count, static_cast<int>(face_files.size()));
    
    return processed_count;
}

/**
 * 预处理单个人脸图像文件
 * @param input_path 输入图像的完整路径
 * @param target_size 目标图像尺寸
 * @param output_path 输出图像的完整路径
 * @return 预处理成功返回true，失败返回false
 */
bool FacePreprocessUtils::preprocessSingleFace(const std::string& input_path,
                                              int target_size,
                                              const std::string& output_path) {
    // 读取图像
    cv::Mat face_image = cv::imread(input_path);
    if (face_image.empty()) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, FACE_PREPROCESS_TAG, 
                   "Failed to load image: %{public}s", input_path.c_str());
        return false;
    }
    
    // 创建预处理器
    FacePreprocessUtils preprocessor(face_image.cols, face_image.rows, target_size);
    
    // 进行预处理
    auto preprocessed_image = preprocessor.convert(face_image);
    if (preprocessed_image == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, FACE_PREPROCESS_TAG, 
                   "Failed to preprocess image: %{public}s", input_path.c_str());
        return false;
    }
    
    // 保存预处理后的图像
    if (!cv::imwrite(output_path, *preprocessed_image)) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, FACE_PREPROCESS_TAG, 
                   "Failed to save preprocessed image: %{public}s", output_path.c_str());
        return false;
    }
    
    // 记录预处理信息
    const letterbox_t& letterbox = preprocessor.getLetterBox();
    OH_LOG_Print(LOG_APP, LOG_DEBUG, GLOBAL_RESMGR, FACE_PREPROCESS_TAG, 
               "Preprocessed %{public}s: scale=%{public}f, x_pad=%{public}d, y_pad=%{public}d", 
               input_path.c_str(), letterbox.scale, letterbox.x_pad, letterbox.y_pad);
    
    return true;
}
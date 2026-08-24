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
#include "face_crop_utils.h"
#include "face_preprocess_utils.h"
#include "hilog/log.h"
#include <algorithm>
#include <sstream>
#include <chrono>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

const int GLOBAL_RESMGR = 0xFF00;
const char *FACE_CROP_TAG = "[FaceCropUtils]";

int FaceCropUtils::cropAndSaveFaces(const cv::Mat& original_image, 
                                   const detect_result_group_t& detect_results,
                                   const std::string& output_dir,
                                   const std::string& input_image_path) {
    if (original_image.empty()) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, FACE_CROP_TAG, "Original image is empty");
        return 0;
    }
    
    int face_count = 0;
    int img_width = original_image.cols;
    int img_height = original_image.rows;
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, FACE_CROP_TAG, 
                "Processing %{public}d detections from image %{public}dx%{public}d", 
                detect_results.count, img_width, img_height);
    
    for (int i = 0; i < detect_results.count; i++) {
        const detect_result_t* det_result = &(detect_results.results[i]);
        
        // 只处理人脸检测结果
        if (strcmp(det_result->name, "face") == 0 && det_result->prop > 0.5) {
            // 调整检测框到图像范围内
            BOX_RECT adjusted_box = clampBoxToImage(det_result->box, img_width, img_height);
            
            // 验证检测框是否有效
            if (isValidBox(adjusted_box, img_width, img_height)) {
                face_count++;
                
                // 生成输出文件路径
                std::string face_filename;
                
                // 使用时间戳命名
                auto now = std::chrono::system_clock::now();
                auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
                face_filename = output_dir + "/face" + std::to_string(face_count) + "_" + std::to_string(timestamp) + ".jpg";
                
                // 清理旧的人脸图片文件，保持每个人脸最多5张
                cleanupOldFaceFiles(output_dir, face_count, 5);
                
                // 裁切并保存人脸
                if (cropSingleFace(original_image, adjusted_box, face_filename)) {
                    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, FACE_CROP_TAG, 
                               "Successfully saved face %{public}d to %{public}s", 
                               face_count, face_filename.c_str());
                    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, FACE_CROP_TAG, 
                               "Face region: (%{public}d,%{public}d) to (%{public}d,%{public}d), confidence: %{public}f", 
                               adjusted_box.left, adjusted_box.top, adjusted_box.right, adjusted_box.bottom, det_result->prop);
                    
                    // 对裁切的人脸图像进行预处理
                    std::string preprocessed_filename;
                    
                    // 使用原有的命名方式
                    size_t dot_pos = face_filename.find_last_of(".");
                    std::string base_name = face_filename.substr(0, dot_pos);
                    preprocessed_filename = base_name + "_preprocessed.jpg";
                    
                    // 使用224x224作为标准的人脸识别输入尺寸
                    if (FacePreprocessUtils::preprocessSingleFace(face_filename, 224, preprocessed_filename)) {
                        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, FACE_CROP_TAG, 
                                   "Successfully preprocessed face %{public}d: %{public}s", 
                                   face_count, preprocessed_filename.c_str());
                    } else {
                        OH_LOG_Print(LOG_APP, LOG_WARN, GLOBAL_RESMGR, FACE_CROP_TAG, 
                                   "Failed to preprocess face %{public}d: %{public}s", 
                                   face_count, face_filename.c_str());
                    }
                } else {
                    OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, FACE_CROP_TAG, 
                               "Failed to save face %{public}d to %{public}s", 
                               face_count, face_filename.c_str());
                    face_count--; // 失败时减少计数
                }
            } else {
                OH_LOG_Print(LOG_APP, LOG_WARN, GLOBAL_RESMGR, FACE_CROP_TAG, 
                           "Invalid face detection box: (%{public}d,%{public}d) to (%{public}d,%{public}d)", 
                           det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom);
            }
        }
    }
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, FACE_CROP_TAG, 
               "Face cropping completed. Total faces saved: %{public}d", face_count);
    
    return face_count;
}

bool FaceCropUtils::cropSingleFace(const cv::Mat& original_image,
                                  const BOX_RECT& face_box,
                                  const std::string& output_path) {
    try {
        // 参考RknnPool.cpp中的face_recognition函数实现
        // 创建裁切区域
        cv::Rect crop_rect(face_box.left, face_box.top, 
                          face_box.right - face_box.left, 
                          face_box.bottom - face_box.top);
        
        // 裁切人脸区域
        cv::Mat crop_img = original_image(crop_rect);
        
        if (crop_img.empty()) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, FACE_CROP_TAG, 
                       "Cropped image is empty for region (%{public}d,%{public}d,%{public}d,%{public}d)", 
                       face_box.left, face_box.top, face_box.right, face_box.bottom);
            return false;
        }
        
        // 保存裁切后的人脸图像
        bool save_success = cv::imwrite(output_path, crop_img);
        
        if (save_success) {
            OH_LOG_Print(LOG_APP, LOG_DEBUG, GLOBAL_RESMGR, FACE_CROP_TAG, 
                       "Face image saved: %{public}s, size: %{public}dx%{public}d", 
                       output_path.c_str(), crop_img.cols, crop_img.rows);
        } else {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, FACE_CROP_TAG, 
                       "Failed to write image to %{public}s", output_path.c_str());
        }
        
        return save_success;
        
    } catch (const cv::Exception& e) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, FACE_CROP_TAG, 
                   "OpenCV exception in cropSingleFace: %{public}s", e.what());
        return false;
    } catch (const std::exception& e) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, FACE_CROP_TAG, 
                   "Exception in cropSingleFace: %{public}s", e.what());
        return false;
    }
}

bool FaceCropUtils::isValidBox(const BOX_RECT& box, int image_width, int image_height) {
    // 检查坐标是否在图像范围内
    if (box.left < 0 || box.top < 0 || 
        box.right > image_width || box.bottom > image_height) {
        return false;
    }
    
    // 检查宽度和高度是否有效
    int width = box.right - box.left;
    int height = box.bottom - box.top;
    
    if (width <= 0 || height <= 0) {
        return false;
    }
    
    // 检查最小尺寸（避免过小的检测框）
    const int MIN_FACE_SIZE = 20;
    if (width < MIN_FACE_SIZE || height < MIN_FACE_SIZE) {
        OH_LOG_Print(LOG_APP, LOG_WARN, GLOBAL_RESMGR, FACE_CROP_TAG, 
                   "Face too small: %{public}dx%{public}d, minimum size: %{public}d", 
                   width, height, MIN_FACE_SIZE);
        return false;
    }
    
    return true;
}

BOX_RECT FaceCropUtils::clampBoxToImage(const BOX_RECT& box, int image_width, int image_height) {
    BOX_RECT clamped_box;
    
    // 确保坐标在图像范围内
    clamped_box.left = std::max(0, box.left);
    clamped_box.top = std::max(0, box.top);
    clamped_box.right = std::min(image_width, box.right);
    clamped_box.bottom = std::min(image_height, box.bottom);
    
    // 确保right > left 和 bottom > top
    if (clamped_box.right <= clamped_box.left) {
        clamped_box.right = clamped_box.left + 1;
    }
    if (clamped_box.bottom <= clamped_box.top) {
        clamped_box.bottom = clamped_box.top + 1;
    }
    
    return clamped_box;
}

void FaceCropUtils::cleanupOldFaceFiles(const std::string& output_dir, int face_index, int max_files) {
    try {
        std::vector<std::string> face_files;
        std::string face_prefix = "face" + std::to_string(face_index) + "_";
        
        // 使用传统的C函数遍历目录
        DIR* dir = opendir(output_dir.c_str());
        if (dir == nullptr) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, FACE_CROP_TAG, 
                       "Failed to open directory: %{public}s", output_dir.c_str());
            return;
        }
        
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            
            // 检查是否是匹配的人脸文件
            if (filename.find(face_prefix) == 0 && filename.find(".jpg") != std::string::npos) {
                std::string full_path = output_dir + "/" + filename;
                
                // 检查是否是普通文件
                struct stat file_stat;
                if (stat(full_path.c_str(), &file_stat) == 0 && S_ISREG(file_stat.st_mode)) {
                    face_files.push_back(full_path);
                }
            }
        }
        closedir(dir);
        
        // 按文件名排序（时间戳在文件名中，所以按文件名排序就是按时间排序）
        std::sort(face_files.begin(), face_files.end());
        
        // 如果文件数量超过限制，删除最旧的文件
        while (face_files.size() >= max_files) {
            std::string oldest_file = face_files.front();
            if (unlink(oldest_file.c_str()) == 0) {
                OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, FACE_CROP_TAG, 
                           "Deleted old face file: %{public}s", oldest_file.c_str());
            } else {
                OH_LOG_Print(LOG_APP, LOG_WARN, GLOBAL_RESMGR, FACE_CROP_TAG, 
                           "Failed to delete old face file: %{public}s", oldest_file.c_str());
            }
            face_files.erase(face_files.begin());
        }
        
    } catch (const std::exception& e) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, FACE_CROP_TAG, 
                   "Error cleaning up old face files: %{public}s", e.what());
    }
}

bool FaceCropUtils::FaceisNumberedImage(const std::string& filename) {
    // 检查文件是否以.jpg结尾
    if (filename.length() < 5 || filename.substr(filename.length() - 4) != ".jpg") {
        return false;
    }
    
    // 检查去掉.jpg后缀的部分是否全为数字
    std::string name_part = filename.substr(0, filename.length() - 4);
    if (name_part.empty()) {
        return false;
    }
    
    for (char c : name_part) {
        if (!std::isdigit(c)) {
            return false;
        }
    }
    
    return true;
}

std::string FaceCropUtils::extractNumberFromFilename(const std::string& filename) {
    // 返回去掉.jpg后缀的部分
    return filename.substr(0, filename.length() - 4);
}

int FaceCropUtils::preprocessNumberedImages(const std::string& input_dir, const std::string& output_dir) {
    int processed_count = 0;
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, FACE_CROP_TAG, 
                "Starting preprocessing of numbered images in directory: %{public}s", input_dir.c_str());
    
    try {
        // 使用传统的C函数遍历目录
        DIR* dir = opendir(input_dir.c_str());
        if (dir == nullptr) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, FACE_CROP_TAG, 
                       "Failed to open input directory: %{public}s", input_dir.c_str());
            return 0;
        }
        
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            
            // 检查是否是数字命名的图片文件
            if (FaceisNumberedImage(filename)) {
                std::string input_path = input_dir + "/" + filename;
                
                // 检查是否是普通文件
                struct stat file_stat;
                if (stat(input_path.c_str(), &file_stat) == 0 && S_ISREG(file_stat.st_mode)) {
                    // 生成预处理后的文件名
                    std::string number = extractNumberFromFilename(filename);
                    std::string output_path = output_dir + "/" + number + "_preprocessed.jpg";
                    
                    // 使用224x224作为标准的人脸识别输入尺寸进行预处理
                    if (FacePreprocessUtils::preprocessSingleFace(input_path, 224, output_path)) {
                        processed_count++;
                        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, FACE_CROP_TAG, 
                                   "Successfully preprocessed %{public}s -> %{public}s", 
                                   filename.c_str(), output_path.c_str());
                    } else {
                        OH_LOG_Print(LOG_APP, LOG_WARN, GLOBAL_RESMGR, FACE_CROP_TAG, 
                                   "Failed to preprocess %{public}s", filename.c_str());
                    }
                }
            }
        }
        closedir(dir);
        
    } catch (const std::exception& e) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, FACE_CROP_TAG, 
                   "Error preprocessing numbered images: %{public}s", e.what());
    }
    
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, FACE_CROP_TAG, 
               "Preprocessing completed. Total images processed: %{public}d", processed_count);
    
    return processed_count;
}
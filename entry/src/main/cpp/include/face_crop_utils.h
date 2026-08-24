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
#ifndef FACE_CROP_UTILS_H
#define FACE_CROP_UTILS_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include <string>

// 前向声明
class FacePreprocessUtils;

/**
 * @brief 人脸裁切工具类
 * 参考RknnPool.cpp中的face_recognition函数实现
 */
class FaceCropUtils {
public:
    /**
     * @brief 根据检测结果裁切人脸区域并保存
     * @param original_image 原始图像
     * @param detect_results 检测结果
     * @param output_dir 输出目录
     * @return 成功裁切的人脸数量
     */
    static int cropAndSaveFaces(const cv::Mat& original_image, 
                               const detect_result_group_t& detect_results,
                               const std::string& output_dir,
                               const std::string& input_image_path = "");
    
    /**
     * @brief 裁切单个人脸区域
     * @param original_image 原始图像
     * @param face_box 人脸检测框
     * @param output_path 输出路径
     * @return 是否成功
     */
    static bool cropSingleFace(const cv::Mat& original_image,
                              const BOX_RECT& face_box,
                              const std::string& output_path);
    
    /**
     * @brief 验证检测框是否有效
     * @param box 检测框
     * @param image_width 图像宽度
     * @param image_height 图像高度
     * @return 是否有效
     */
    static bool isValidBox(const BOX_RECT& box, int image_width, int image_height);
    
    /**
     * @brief 清理旧的人脸图片文件
     * @param output_dir 输出目录
     * @param face_index 人脸索引
     * @param max_files 每个人脸最大保留文件数
     */
    static void cleanupOldFaceFiles(const std::string& output_dir, int face_index, int max_files);
    
    /**
     * @brief 调整检测框坐标到图像范围内
     * @param box 检测框
     * @param image_width 图像宽度
     * @param image_height 图像高度
     * @return 调整后的检测框
     */
    static BOX_RECT clampBoxToImage(const BOX_RECT& box, int image_width, int image_height);
    
    /**
     * @brief 检查文件名是否为数字命名的图片（如1.jpg, 2.jpg等）
     * @param filename 文件名
     * @return 是否为数字命名的图片
     */
    static bool FaceisNumberedImage(const std::string& filename);
    
    /**
     * @brief 从数字命名的文件中提取数字部分
     * @param filename 文件名
     * @return 数字部分字符串
     */
    static std::string extractNumberFromFilename(const std::string& filename);
    
    /**
     * @brief 预处理沙箱中的数字命名图片（1.jpg, 2.jpg等）
     * @param input_dir 输入目录路径
     * @param output_dir 输出目录路径
     * @return 成功处理的图片数量
     */
    static int preprocessNumberedImages(const std::string& input_dir, const std::string& output_dir);
};

#endif // FACE_CROP_UTILS_H
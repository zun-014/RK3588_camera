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
#ifndef FACE_PREPROCESS_UTILS_H
#define FACE_PREPROCESS_UTILS_H

#include "Common.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <string>
#include <memory>

/**
 * @brief 人脸图像预处理工具类
 * 参考ImageProcess.cpp中的预处理方法实现
 */
class FacePreprocessUtils {
public:
    /**
     * @brief 构造函数，计算缩放比例和填充大小
     * @param width 原始图像宽度
     * @param height 原始图像高度
     * @param target_size 目标尺寸
     */
    FacePreprocessUtils(int width, int height, int target_size);
    
    /**
     * @brief 将图像转换为目标大小，填充并返回
     * @param src 源图像
     * @return 预处理后的图像
     */
    std::unique_ptr<cv::Mat> convert(const cv::Mat& src);
    
    /**
     * @brief 获取letterbox配置信息
     * @return letterbox配置
     */
    const letterbox_t& getLetterBox() const;
    
    /**
     * @brief 批量预处理人脸图像文件
     * @param face_files_dir 人脸图像文件目录
     * @param target_size 目标尺寸
     * @param output_dir 输出目录
     * @return 成功预处理的文件数量
     */
    static int preprocessFaceImages(const std::string& face_files_dir, 
                                   int target_size, 
                                   const std::string& output_dir);
    
    /**
     * @brief 预处理单个人脸图像文件
     * @param input_path 输入文件路径
     * @param target_size 目标尺寸
     * @param output_path 输出文件路径
     * @return 是否成功
     */
    static bool preprocessSingleFace(const std::string& input_path,
                                     int target_size,
                                     const std::string& output_path);

private:
    double scale_;           // 缩放比例
    int padding_x_;          // X方向填充大小
    int padding_y_;          // Y方向填充大小
    cv::Size new_size_;      // 缩放后的尺寸
    int target_size_;        // 目标尺寸
    letterbox_t letterbox_;  // letterbox配置信息
};

#endif // FACE_PREPROCESS_UTILS_H
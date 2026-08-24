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
#ifndef _RKNN_YOLOV5_DEMO_PREPROCESS_H_
#define _RKNN_YOLOV5_DEMO_PREPROCESS_H_

#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"

void letterbox(const cv::Mat &image, cv::Mat &padded_image, BOX_RECT &pads, 
               const float scale, const cv::Size &target_size, 
               const cv::Scalar &pad_color = cv::Scalar(128, 128, 128));

#endif // _RKNN_YOLOV5_DEMO_PREPROCESS_H_

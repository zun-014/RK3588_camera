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
#ifndef _RKNN_YOLOV5_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV5_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include <rawfile/raw_file_manager.h>

#define YOLOV5_OBJ_NAME_MAX_SIZE 16
#define YOLOV5_OBJ_NUMB_MAX_SIZE 64
#define YOLOV5_OBJ_CLASS_NUM 11
#define YOLOV5_NMS_THRESH 0.45
#define YOLOV5_BOX_THRESH 0.25
#define YOLOV5_PROP_BOX_SIZE (5 + YOLOV5_OBJ_CLASS_NUM)

typedef struct _BOX_RECT {
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct _detect_result_t {
    char name[YOLOV5_OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float prop;
    int landmarks[10]; // Store 5 keypoints' x,y coordinates
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[YOLOV5_OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, BOX_RECT pads, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);



void deinitPostProcess();
#endif //_RKNN_YOLOV5_DEMO_POSTPROCESS_H_

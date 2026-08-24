/****************************************************************************
 *
 *    Copyright (c) 2017 - 2025 by Rockchip Corp.  All rights reserved.
 *
 *    The material in this file is confidential and contains trade secrets
 *    of Rockchip Corporation. This is proprietary information owned by
 *    Rockchip Corporation. No part of this work may be disclosed,
 *    reproduced, copied, transmitted, or used in any way for any purpose,
 *    without the express written permission of Rockchip Corporation.
 *
 *****************************************************************************/

#pragma once
#include <stdint.h>
#include <string>

#include "Common.hpp"
#include "rknn/rknn_api.h"

int retinaface_post_process(rknn_app_context_t *app_ctx, rknn_output *outputs,
                 letterbox_t *letter_box, retinaface_result *results);

const char *coco_cls_to_name(int cls_id);

int init_yolo_post_process(const char *label_path);

void deinit_yolo_post_process();

int yolo_post_process(rknn_app_context_t *app_ctx, rknn_output *outputs,
                 letterbox_t *letter_box, float conf_threshold,
                 float nms_threshold, yolo_result_list *results);
                      

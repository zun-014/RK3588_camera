/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_IMGPROC_HAL_INTERFACE_H
#define OPENCV_IMGPROC_HAL_INTERFACE_H

//! @addtogroup imgproc_hal_interface
//! @{

//! @name Interpolation modes
//! @sa cv::InterpolationFlags
//! @{
#define CV_HAL_INTER_NEAREST 0
#define CV_HAL_INTER_LINEAR 1
#define CV_HAL_INTER_CUBIC 2
#define CV_HAL_INTER_AREA 3
#define CV_HAL_INTER_LANCZOS4 4
//! @}

//! @name Morphology operations
//! @sa cv::MorphTypes
//! @{
#define CV_HAL_MORPH_ERODE 0
#define CV_HAL_MORPH_DILATE 1
//! @}

//! @name Threshold types
//! @sa cv::ThresholdTypes
//! @{
#define CV_HAL_THRESH_BINARY      0
#define CV_HAL_THRESH_BINARY_INV  1
#define CV_HAL_THRESH_TRUNC       2
#define CV_HAL_THRESH_TOZERO      3
#define CV_HAL_THRESH_TOZERO_INV  4
#define CV_HAL_THRESH_MASK        7
#define CV_HAL_THRESH_OTSU        8
#define CV_HAL_THRESH_TRIANGLE    16
//! @}

//! @name Adaptive threshold algorithm
//! @sa cv::AdaptiveThresholdTypes
//! @{
#define CV_HAL_ADAPTIVE_THRESH_MEAN_C     0
#define CV_HAL_ADAPTIVE_THRESH_GAUSSIAN_C 1
//! @}

//! @}

#endif

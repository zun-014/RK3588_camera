# the minimum version of CMake.
cmake_minimum_required(VERSION 3.5.0)
project(Native_Camera)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED PACKAGE_FIND_FILE)
    include(${PACKAGE_FIND_FILE})
endif()

include_directories(${NATIVERENDER_ROOT_PATH}
                    ${NATIVERENDER_ROOT_PATH}/include
                    ${NATIVERENDER_ROOT_PATH}/types/include
                    ${NATIVERENDER_ROOT_PATH}/include/opencv2
                    ${NATIVERENDER_ROOT_PATH}/include/opencv2/core)

link_libraries( ${NATIVERENDER_ROOT_PATH}/../../../libs/${OHOS_ARCH}/librknnrt.so)

add_library(entry SHARED facial_recognition.cpp postprocess.cc face_crop_utils.cpp face_preprocess_utils.cpp model.cpp)

target_link_libraries(entry PUBLIC
        libace_napi.z.so
        libhilog_ndk.z.so
        librawfile.z.so
        ${NATIVERENDER_ROOT_PATH}/../../../libs/${OHOS_ARCH}/libopencv_core.so
        ${NATIVERENDER_ROOT_PATH}/../../../libs/${OHOS_ARCH}/libopencv_imgcodecs.so
        ${NATIVERENDER_ROOT_PATH}/../../../libs/${OHOS_ARCH}/libopencv_imgproc.so)
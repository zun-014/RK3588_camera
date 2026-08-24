# Change_model_RK3568

## 切换模型需注释代码

路径：src/main/cpp/facial_recognition.cpp

```C
/*
    src/main/cpp/facial_recognition.cpp 第649-792行
*/
//        // 使用FaceCropUtils进行人脸裁切
//        int face_count = FaceCropUtils::cropAndSaveFaces(orig_img, detect_result_group, std::string(sandboxDir), imagePath);
//        
//        // 启动异步人脸识别工作线程（如果尚未启动）
//        start_face_recognition_worker();
//        
//        // 对刚裁切生成的人脸进行异步识别对比（只处理刚生成的图片）
//        if (face_count > 0) {
//            try {
//                DIR* dir = opendir(sandboxDir);
//                if (dir != nullptr) {
//                    struct dirent* entry;
//                    std::vector<std::pair<std::string, time_t>> face_files_with_time;
//                    
//                    // 获取当前时间戳，用于筛选刚生成的文件
//                    time_t current_time = time(nullptr);
//                    
//                    // 查找face*_preprocessed.jpg文件
//                    while ((entry = readdir(dir)) != nullptr) {
//                        std::string filename = entry->d_name;
//                        // 查找face开头且包含_preprocessed.jpg的文件
//                        if (filename.find("face") == 0 && filename.find("_preprocessed.jpg") != std::string::npos) {
//                            
//                            // 获取文件的修改时间
//                            std::string full_path = std::string(sandboxDir) + "/" + filename;
//                            struct stat file_stat;
//                            if (stat(full_path.c_str(), &file_stat) == 0) {
//                                // 只处理最近1秒内创建/修改的文件（刚裁切生成的）
//                                if (current_time - file_stat.st_mtime <= 1) {
//                                    face_files_with_time.push_back({filename, file_stat.st_mtime});
//                                }
//                            }
//                        }
//                    }
//                    closedir(dir);
//                    
//                    // 按修改时间排序，优先处理最新的文件
//                    std::sort(face_files_with_time.begin(), face_files_with_time.end(), 
//                             [](const std::pair<std::string, time_t>& a, const std::pair<std::string, time_t>& b) {
//                                 return a.second > b.second;
//                             });
//                    
//                    // 将刚生成的人脸文件添加到异步处理队列，限制为face_count数量
//                    for (size_t i = 0; i < face_files_with_time.size() && i < static_cast<size_t>(face_count); i++) {
//                        const std::string& face_filename = face_files_with_time[i].first;
//                        std::string face_path = std::string(sandboxDir) + "/" + face_filename;
//                        add_face_recognition_task(face_path, std::string(sandboxDir), mNativeResMgr);
//                        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Added face recognition task for newly processed file: %{public}s", face_filename.c_str());
//                    }
//                }
//            } catch (const std::exception& e) {
//                OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Exception while adding face recognition tasks: %{public}s", e.what());
//            }
//        }
//        
//        // 获取最新的异步识别结果（如果有的话）
//        float similarity;
//        bool is_same_person, has_result;
//        std::string processed_file, matched_reference_path;
//        
//        if (get_latest_face_recognition_result(similarity, is_same_person, has_result, processed_file, matched_reference_path)) {
//            // 添加相似度到返回结果
//            napi_value similarity_value;
//            napi_create_double(env, similarity, &similarity_value);
//            napi_set_named_property(env, result, "faceSimilarity", similarity_value);
//            
//            // 添加是否为同一人的判断
//            napi_value is_same_person_value;
//            napi_get_boolean(env, is_same_person, &is_same_person_value);
//            napi_set_named_property(env, result, "isSamePerson", is_same_person_value);
//            
//            // 添加匹配的参考人脸路径
//            if (!matched_reference_path.empty()) {
//                napi_value matched_reference_path_value;
//                napi_create_string_utf8(env, matched_reference_path.c_str(), NAPI_AUTO_LENGTH, &matched_reference_path_value);
//                napi_set_named_property(env, result, "matchedReferencePath", matched_reference_path_value);
//            }
//            
//            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Using latest face recognition result: %{public}s (similarity: %{public}f)", 
//                       is_same_person ? "Same person" : "Different person", similarity);
//        }
```

路径：src/main/cpp/facial_recognition.cpp

```C
/*
    src/main/cpp/facial_recognition.cpp 第766-796行
*/
//        // 添加人脸图片路径到返回结果 - 动态读取带时间戳的文件
//        if (face_count > 0) {
//            DIR* face_dir = opendir(sandboxDir);
//            if (face_dir != nullptr) {
//                struct dirent* entry;
//                std::vector<std::string> face_files;
//                
//                // 收集所有人脸文件
//                while ((entry = readdir(face_dir)) != nullptr) {
//                    std::string filename = entry->d_name;
//                    if (filename.find("face") == 0 && filename.find(".jpg") != std::string::npos) {
//                        face_files.push_back(std::string(sandboxDir) + "/" + filename);
//                    }
//                }
//                closedir(face_dir);
//                
//                // 按文件名排序以确保顺序一致
//                std::sort(face_files.begin(), face_files.end());
//                
//                // 添加到返回结果
//                for (size_t i = 0; i < face_files.size() && i < static_cast<size_t>(face_count); i++) {
//                    napi_value face_path_value;
//                    napi_create_string_utf8(env, face_files[i].c_str(), NAPI_AUTO_LENGTH, &face_path_value);
//                    napi_set_element(env, faceImagePaths, i, face_path_value);
//                }
//            }
//        }
//        
//        if (face_count > 0) {
//            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Successfully processed and saved %{public}d face(s) using FaceCropUtils", face_count);
//        }
```
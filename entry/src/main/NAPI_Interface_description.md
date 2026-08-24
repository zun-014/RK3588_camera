# NAPI Interface description

## 概述

本文档基于 `Index.d.ts` 文件中定义的TypeScript接口和 `Index.ets` 文件中的实际调用示例，详细说明RKNN人脸识别NAPI接口的功能、参数和调用方式。

## 接口列表

### 1. processImageWithRKNN - 图像处理与人脸识别

#### 接口定义

```typescript
export const processImageWithRKNN: (
  photoUri: string, 
  resMgr: resourceManager.ResourceManager, 
  sandboxDir: string
) => Promise<RknnProcessResult>;
```

#### 功能描述

使用RKNN模型对输入图像进行人脸检测和识别，包括：

- 人脸检测和定位
- 人脸特征提取
- 人脸识别和相似度计算
- 与数据库中的参考人脸进行匹配

#### 参数说明

- `photoUri`: 图片URI路径，支持file://协议
- `resMgr`: 资源管理器，用于访问应用资源
- `sandboxDir`: 沙箱目录路径，用于保存处理结果

#### 返回值

返回 `Promise<RknnProcessResult>` 对象，包含以下字段：

- `success`: 处理是否成功
- `message`: 处理结果消息
- `savedImagePath`: 保存的处理后图片路径
- `imageInfo`: 图像信息（宽度、高度、通道数）
- `inferenceTime`: 推理时间（毫秒）
- `detections`: 检测结果数组
- `faceImagePaths`: 裁切的人脸图片路径数组
- `faceSimilarity`: 人脸相似度（0-1之间）
- `isSamePerson`: 是否为同一人
- `matchedReferencePath`: 匹配的参考人脸图片路径

#### 调用示例

```typescript
// 在Index.ets中的调用方式
private async processImageWithInference(photoUri: string): Promise<void> {
  try {
    const result: RknnProcessResult = 
      await rknn.processImageWithRKNN(
        photoUri, 
        getContext(this).resourceManager, 
        getContext(this).filesDir
      );
  
    if (result.success) {
      // 处理成功，更新UI状态
      this.detections = result.detections || [];
      this.imageInfo = result.imageInfo || null;
      this.faceImagePaths = result.faceImagePaths || [];
      this.faceSimilarity = result.faceSimilarity || 0;
      this.isSamePerson = result.isSamePerson || false;
      this.matchedReferencePath = result.matchedReferencePath || '';
      this.imgThumbnail = 'file://' + result.savedImagePath;
  
      Logger.info(this.tag, `推理时间: ${result.inferenceTime}ms`);
      Logger.info(this.tag, `检测到${this.detections.length}个人脸`);
    } else {
      Logger.error(this.tag, 'RKNN处理失败:', result.message);
    }
  } catch (error) {
    Logger.error(this.tag, '调用RKNN处理接口失败:', error.message);
  }
}
```

### 2. saveModelFile - 保存模型文件

#### 接口定义

```typescript
export const saveModelFile: (
  resMgr: resourceManager.ResourceManager, 
  sandboxDir: string
) => SaveModelResult;
```

#### 功能描述

将RKNN模型文件从应用资源目录复制到沙箱目录，用于后续的推理计算。

#### 参数说明

- `resMgr`: 资源管理器，用于访问应用资源
- `sandboxDir`: 沙箱目录路径，模型文件的保存位置

#### 返回值

返回 `SaveModelResult` 对象，包含以下字段：

- `success`: 保存是否成功
- `message`: 保存结果消息
- `savedModelPath`: 保存的模型文件路径（仅在成功时存在）

#### 调用示例

```typescript
// 在Index.ets的aboutToAppear()方法中调用
async aboutToAppear() {
  try {
    Logger.info(this.tag, '应用启动时自动保存模型文件');
    const result: SaveModelResult = rknn.saveModelFile(
      getContext(this).resourceManager, 
      getContext(this).filesDir
    );
  
    if (result.success) {
      Logger.info(this.tag, `模型文件保存成功: ${result.message}`);
      Logger.info(this.tag, `保存路径: ${result.savedModelPath}`);
      prompt.showToast({
        message: `模型文件保存成功`,
        duration: 2000
      });
    } else {
      Logger.error(this.tag, `模型文件保存失败: ${result.message}`);
      prompt.showToast({
        message: `模型文件保存失败: ${result.message}`,
        duration: 3000
      });
    }
  } catch (error) {
    Logger.error(this.tag, `调用保存模型文件接口失败: ${error}`);
  }
}
```

### 3. saveFaceDatabaseImages - 保存人脸数据库图片

#### 接口定义

```typescript
export const saveFaceDatabaseImages: (
  resMgr: resourceManager.ResourceManager, 
  sandboxDir: string
) => FaceDatabaseSaveResult;
```

#### 功能描述

将人脸数据库图片从应用资源目录复制到沙箱目录，用于人脸识别的参考对比。

#### 参数说明

- `resMgr`: 资源管理器，用于访问应用资源
- `sandboxDir`: 沙箱目录路径，人脸数据库图片的保存位置

#### 返回值

返回 `FaceDatabaseSaveResult` 对象，包含以下字段：

- `success`: 保存是否成功
- `message`: 保存结果消息
- `savedCount`: 成功保存的图片数量
- `totalAttempted`: 尝试保存的图片总数

#### 调用示例

```typescript
// 在Index.ets的aboutToAppear()方法中调用
async aboutToAppear() {
  try {
    Logger.info(this.tag, '应用启动时自动保存人脸数据库图片');
    const faceDbResult: FaceDatabaseSaveResult = rknn.saveFaceDatabaseImages(
      getContext(this).resourceManager, 
      getContext(this).filesDir
    );
  
    if (faceDbResult.success) {
      Logger.info(this.tag, `人脸数据库图片保存成功: ${faceDbResult.message}`);
      Logger.info(this.tag, `成功保存: ${faceDbResult.savedCount}/${faceDbResult.totalAttempted} 张图片`);
      prompt.showToast({
        message: `人脸数据库图片保存成功: ${faceDbResult.savedCount}张`,
        duration: 2000
      });
    } else {
      Logger.error(this.tag, `人脸数据库图片保存失败: ${faceDbResult.message}`);
      prompt.showToast({
        message: `人脸数据库图片保存失败`,
        duration: 3000
      });
    }
  } catch (error) {
    Logger.error(this.tag, `调用保存人脸数据库图片接口失败: ${error}`);
  }
}
```

## 数据结构说明

### Detection - 检测结果

```typescript
interface Detection {
  name: string;       // 检测对象名称
  confidence: number; // 置信度（0-1之间）
  left: number;       // 左边界坐标
  top: number;        // 上边界坐标
  right: number;      // 右边界坐标
  bottom: number;     // 下边界坐标
}
```

### ImageInfo - 图像信息

```typescript
interface ImageInfo {
  width: number;    // 图像宽度
  height: number;   // 图像高度
  channel: number;  // 图像通道数
}
```

### RknnTensorAttr - Tensor属性

```typescript
interface RknnTensorAttr {
  index: number;      // tensor索引
  name: string;       // tensor名称
  n_dims: number;     // 维度数量
  dims: number[];     // 维度信息
  n_elems: number;    // 元素数量
  size: number;       // 大小
  format: string;     // 格式
  type: string;       // 类型
  qnt_type: string;   // 量化类型
}
```

## 使用流程

### 1. 应用初始化阶段

```typescript
// 在aboutToAppear()生命周期中
// 1. 保存模型文件
const modelResult = rknn.saveModelFile(resourceManager, filesDir);

// 2. 保存人脸数据库图片
const faceDbResult = rknn.saveFaceDatabaseImages(resourceManager, filesDir);
```

### 2. 图像处理阶段

```typescript
// 在拍照回调中
this.cameraService.setTakePictureCallback((photoUri: string) => {
  // 使用流水线处理机制，每N张图片推理一次
  this.processImagePipeline(photoUri);
});

// 推理处理
const result = await rknn.processImageWithRKNN(photoUri, resourceManager, filesDir);
```

### 3. 结果处理阶段

```typescript
if (result.success) {
  // 更新UI显示检测框
  this.detections = result.detections;
  
  // 显示人脸识别结果
  this.faceSimilarity = result.faceSimilarity;
  this.isSamePerson = result.isSamePerson;
  this.matchedReferencePath = result.matchedReferencePath;
}
```

## 注意事项

1. `processImageWithRKNN` 是异步接口，需要使用 `await` 等待结果
2. `saveModelFile` 和 `saveFaceDatabaseImages` 是同步接口，建议在应用启动时调用
3. 所有接口都需要传入有效的资源管理器和沙箱目录路径
4. 推理结果中的可选字段需要进行空值检查
5. 建议在调用前检查文件路径的有效性

## 错误处理

所有接口都应该包含完善的错误处理机制：

```typescript
try {
  const result = await rknn.processImageWithRKNN(photoUri, resMgr, sandboxDir);
  if (result.success) {
    // 处理成功逻辑
  } else {
    // 处理失败逻辑
    Logger.error(tag, `处理失败: ${result.message}`);
  }
} catch (error) {
  // 异常处理
  Logger.error(tag, `接口调用异常: ${error.message}`);
}
```

## 开源协议说明

本项目使用了多个开源组件和库，以下是相关的开源协议信息：

### 主要开源组件

#### 1. RKNN SDK

- **组件名称**: RKNN (Rockchip Neural Network) SDK
- **版本**: 根据具体使用版本
- **协议**: Apache License 2.0
- **用途**: 提供神经网络推理引擎，用于人脸检测和识别
- **官方网站**: https://github.com/rockchip-linux/rknn-toolkit2

#### 2. OpenCV

- **组件名称**: OpenCV (Open Source Computer Vision Library)
- **版本**: 根据具体使用版本
- **协议**: Apache License 2.0
- **用途**: 图像处理、计算机视觉算法支持
- **官方网站**: https://opencv.org/

#### 3. OpenHarmony SDK

- **组件名称**: OpenHarmony Native API
- **版本**: 根据具体使用版本
- **协议**: Apache License 2.0
- **用途**: 提供OpenHarmony平台的原生API支持
- **官方网站**: https://www.openharmony.cn/

### 协议兼容性

所有使用的开源组件均采用与Apache License 2.0兼容的协议，确保：

- 可以在商业项目中使用
- 可以进行二次开发和分发
- 遵循相应的协议要求和限制

---

**注意**: 本协议说明仅供参考，具体的协议条款请以各开源组件的官方协议文件为准。在商业使用前，建议咨询法律专业人士以确保合规性。

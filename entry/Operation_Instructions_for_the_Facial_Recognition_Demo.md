# Operation Instructions for the Facial Recognition Demo

## 概述

本文档详细介绍基于OpenHarmony的RKNN人脸识别Demo程序的运行机制、功能特性和技术实现。该Demo程序集成了摄像头预览、人脸检测、人脸识别、相似度计算等功能，采用流水线处理机制优化性能。

## 程序架构

### 核心组件

- **Index.ets**: 主界面组件，负责UI渲染和业务逻辑
- **CameraService**: 摄像头服务，处理摄像头初始化和拍照
- **Logger**: 日志服务，统一管理日志输出
- **RKNN NAPI**: 原生接口，提供人脸检测和识别功能

### 技术栈

- **前端**: OpenHarmony ArkTS/ETS
- **AI推理**: RKNN (Rockchip Neural Network)
- **图像处理**: OpenCV
- **原生接口**: NAPI (Native API)

## 功能特性

### 1. 双栏布局界面

程序采用左右双栏布局设计：

- **左侧**: 摄像头实时预览区域
- **右侧**: 拍摄图片显示和人脸检测结果展示

### 2. 流水线处理机制

为了优化性能和提高实时性，程序实现了智能的流水线处理机制：

```typescript
// 每3张图片推理1张，提高人脸检测实时性
private readonly INFERENCE_INTERVAL: number = 3;
```

**处理流程**:
1. 接收摄像头预览帧
2. 帧计数器递增
3. 判断是否达到推理间隔（每3张推理1张）
4. 需要推理：后台异步执行RKNN推理
5. 不需要推理：直接保存并显示图片

### 3. 人脸检测与识别

#### 检测功能
- 实时人脸检测
- 检测框绘制（绿色边框）
- 置信度显示
- 多人脸同时检测

#### 识别功能
- 人脸特征提取
- 与数据库人脸比对
- 相似度计算
- 同一人判断

### 4. 文件管理

#### 自动文件管理
- 输出图片自动命名（`out_timestamp.jpg`）
- 最多保留20张输出图片
- 超出限制时自动删除最旧的10张

#### 模型文件管理
- 应用启动时自动保存RKNN模型文件
- 自动保存人脸数据库图片
- 文件保存状态提示

## 运行流程

### 1. 应用启动阶段

```typescript
async aboutToAppear() {
  // 1. 初始化摄像头表面ID
  this.surfaceId = this.mXComponentController.getXComponentSurfaceId()
  
  // 2. 自动保存RKNN模型文件
  const result: SaveModelResult = rknn.saveModelFile(...)
  
  // 3. 自动保存人脸数据库图片
  const faceDbResult: FaceDatabaseSaveResult = rknn.saveFaceDatabaseImages(...)
  
  // 4. 设置拍照回调
  this.cameraService.setTakePictureCallback((photoUri: string) => {
    this.processImagePipeline(photoUri);
  })
}
```

### 2. 摄像头初始化

```typescript
async initCamera(): Promise<boolean> {
  const isSuccess = await this.cameraService.initCamera(this.surfaceId)
  return isSuccess
}
```

### 3. 图片处理流水线

#### 流水线入口
```typescript
private async processImagePipeline(photoUri: string): Promise<void> {
  this.frameCounter++;
  const shouldInference = (this.frameCounter % this.INFERENCE_INTERVAL === 0);
  
  if (shouldInference && !this.isInferenceRunning) {
    // 执行RKNN推理
    this.processImageWithInference(photoUri);
  } else {
    // 直接保存显示
    this.saveImageDirectly(photoUri);
  }
}
```

#### RKNN推理处理
```typescript
private async processImageWithInference(photoUri: string): Promise<void> {
  const result: RknnProcessResult = await rknn.processImageWithRKNN(
    photoUri, 
    getContext(this).resourceManager, 
    getContext(this).filesDir
  );
  
  if (result.success) {
    // 更新检测结果
    this.detections = result.detections || [];
    this.imageInfo = result.imageInfo || null;
    this.faceImagePaths = result.faceImagePaths || [];
    this.faceSimilarity = result.faceSimilarity || 0;
    this.isSamePerson = result.isSamePerson || false;
    this.matchedReferencePath = result.matchedReferencePath || '';
    
    // 更新UI显示
    this.imgThumbnail = 'file://' + result.savedImagePath;
  }
}
```

### 4. UI渲染与交互

#### 检测框绘制
```typescript
// 动态绘制人脸检测框
ForEach(this.detections, (detection: Detection, index: number) => {
  Stack() {
    // 检测框
    Rect()
      .width(`${((detection.right - detection.left) / this.imageInfo!.width * 100)}%`)
      .height(`${((detection.bottom - detection.top) / this.imageInfo!.height * 100)}%`)
      .fill(Color.Transparent)
      .stroke(Color.Green)
      .strokeWidth(2)
    
    // 标签文本
    Text(`${detection.name} ${(detection.confidence * 100).toFixed(1)}%`)
      .fontSize(11)
      .fontColor(Color.White)
      .backgroundColor(Color.Green)
  }
})
```

#### 识别结果显示
```typescript
// 显示匹配的参考人脸和相似度
if (this.isSamePerson && this.matchedReferencePath) {
  Image('file://' + this.matchedReferencePath)
  Text(`相似度: ${(this.faceSimilarity * 100).toFixed(1)}%`)
  Text(this.isSamePerson ? '是同一个人' : '不是同一个人')
}
```

## 性能优化策略

### 1. 流水线处理
- **间隔推理**: 每3张图片推理1张，减少计算负载
- **异步处理**: 推理过程不阻塞UI线程
- **状态管理**: 防止重复推理，确保单线程执行

### 2. 内存管理
- **文件数量控制**: 最多保留20张输出图片
- **自动清理**: 超出限制时删除旧文件
- **及时释放**: 处理完成后及时更新状态

### 3. UI优化
- **条件渲染**: 只在有检测结果时绘制检测框
- **唯一标识**: 使用唯一key避免不必要的重绘
- **对象适配**: 图片自适应容器大小

## 数据结构

### 检测结果 (Detection)
```typescript
interface Detection {
  left: number;     // 检测框左边界
  top: number;      // 检测框上边界
  right: number;    // 检测框右边界
  bottom: number;   // 检测框下边界
  confidence: number; // 置信度
  name: string;     // 检测类别名称
}
```

### 图像信息 (ImageInfo)
```typescript
interface ImageInfo {
  width: number;    // 图像宽度
  height: number;   // 图像高度
}
```

### RKNN处理结果 (RknnProcessResult)
```typescript
interface RknnProcessResult {
  success: boolean;              // 处理是否成功
  message: string;               // 处理消息
  savedImagePath: string;        // 保存的图片路径
  imageInfo: ImageInfo;          // 图像信息
  inferenceTime: number;         // 推理时间(ms)
  detections: Detection[];       // 检测结果数组
  faceImagePaths: string[];      // 裁切的人脸图片路径
  faceSimilarity: number;        // 人脸相似度
  isSamePerson: boolean;         // 是否为同一人
  matchedReferencePath: string;  // 匹配的参考人脸路径
}
```

## 状态管理

### 核心状态变量

```typescript
@State rotationAngle: number = 90;           // 摄像头旋转角度
@State imgThumbnail: string = '';            // 显示的图片路径
@State curModel: number = CameraMode.MODE_PHOTO; // 摄像头模式
@State stopRunning: boolean = false;         // 停止运行标志

// 检测结果相关状态
@State detections: Detection[] = [];         // 检测结果数组
@State imageInfo: ImageInfo | null = null;   // 图像信息
@State faceImagePaths: string[] = [];        // 人脸图片路径
@State faceSimilarity: number = 0;           // 人脸相似度
@State isSamePerson: boolean = false;        // 是否同一人
@State matchedReferencePath: string = '';    // 匹配的参考人脸
@State testFaceImagePath: string = '';       // 测试人脸图片路径

// 流水线处理状态
@State frameCounter: number = 0;             // 帧计数器
@State isInferenceRunning: boolean = false;  // 推理运行状态
```

## 错误处理

### 1. RKNN处理错误
```typescript
try {
  const result: RknnProcessResult = await rknn.processImageWithRKNN(...);
  if (result.success) {
    // 处理成功逻辑
  } else {
    Logger.error(this.tag, 'RKNN处理失败:', result.message);
    // 清空检测结果
    this.detections = [];
    this.imageInfo = null;
  }
} catch (error) {
  Logger.error(this.tag, '调用RKNN处理接口失败:', error.message);
  // 清空检测结果
}
```

### 2. 文件操作错误
```typescript
try {
  await fs.copyFile(sourceFile, outputPath);
  Logger.info(this.tag, `直接保存图片成功: ${outputPath}`);
} catch (error) {
  Logger.error(this.tag, `直接保存图片失败: ${error}`);
}
```

### 3. 摄像头初始化错误
```typescript
const isSuccess = await this.cameraService.initCamera(this.surfaceId);
if (!isSuccess) {
  Logger.error(this.tag, '摄像头初始化失败');
}
```

## 日志系统

程序使用统一的Logger服务记录运行状态：

```typescript
Logger.info(this.tag, `收到第${this.frameCounter}张图片: ${photoUri}`);
Logger.info(this.tag, `第${this.frameCounter}张图片，开始RKNN推理`);
Logger.info(this.tag, `检测到${this.detections.length}个人脸，更新框`);
Logger.info(this.tag, `人脸识别结果: 相似度=${this.faceSimilarity}, 是否同一人=${this.isSamePerson}`);
Logger.error(this.tag, 'RKNN处理失败:', result.message);
```

## 使用说明

### 1. 启动应用
- 应用自动初始化摄像头
- 自动保存RKNN模型文件和人脸数据库
- 显示保存结果提示

### 2. 实时预览
- 左侧显示摄像头实时预览
- 摄像头自动旋转90度适配界面

### 3. 人脸检测
- 程序自动进行人脸检测
- 右侧显示检测结果图片
- 绿色框标注检测到的人脸
- 显示检测置信度

### 4. 人脸识别
- 自动与数据库人脸进行比对
- 显示相似度百分比
- 显示是否为同一人的判断结果
- 展示匹配的参考人脸图片

### 5. 图片管理
- 点击图片可查看详情
- 程序自动管理输出文件数量
- 保持最多20张历史图片

## 技术特点

### 1. 高性能
- 流水线处理机制减少计算负载
- 异步推理不阻塞UI
- 智能文件管理避免存储溢出

### 2. 高精度
- 基于RKNN深度学习模型
- 支持多人脸同时检测
- 精确的相似度计算

### 3. 用户友好
- 直观的双栏布局
- 实时的检测结果反馈
- 清晰的识别结果展示

### 4. 稳定可靠
- 完善的错误处理机制
- 详细的日志记录
- 自动资源管理

## 总结

本人脸识别Demo程序展示了在OpenHarmony平台上集成RKNN AI推理引擎的完整解决方案。通过流水线处理机制、异步推理、智能文件管理等技术手段，实现了高性能、高精度的实时人脸检测与识别功能。程序架构清晰、功能完善、用户体验良好，为OpenHarmony AI应用开发提供了优秀的参考示例。
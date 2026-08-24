import { resourceManager } from "@kit.LocalizationKit";

// Tensor属性接口
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

// 图像信息接口
interface ImageInfo {
  width: number;    // 图像宽度
  height: number;   // 图像高度
  channel: number;  // 图像通道数
}

// 检测结果接口
interface Detection {
  name: string;       // 检测对象名称
  confidence: number; // 置信度
  left: number;       // 左边界
  top: number;        // 上边界
  right: number;      // 右边界
  bottom: number;     // 下边界
}

// RKNN模型初始化结果接口
interface RknnInitResult {
  success: boolean;  // 初始化是否成功
  message: string;   // 初始化结果消息
  context?: number;  // 模型上下文（仅在成功时存在）
  sdkVersion?: string; // SDK版本信息（仅在成功时存在）
  ioNum?: {
    inputNum: number;  // 输入tensor个数
    outputNum: number; // 输出tensor个数
  }; // 输入输出tensor个数信息（仅在成功时存在）
  inputAttrs?: RknnTensorAttr[];  // 输入tensor属性数组（仅在成功时存在）
  outputAttrs?: RknnTensorAttr[]; // 输出tensor属性数组（仅在成功时存在）
  imageInfo?: ImageInfo; // 图像信息（仅在成功时存在）
  savedImagePath?: string; // 保存的图片路径（仅在成功时存在）
}

// RKNN图像处理结果接口
interface RknnProcessResult {
  success: boolean;        // 处理是否成功
  message: string;         // 处理结果消息
  savedImagePath?: string; // 保存的处理后图片路径（仅在成功时存在）
  imageInfo?: ImageInfo;   // 图像信息（仅在成功时存在）
  inferenceTime?: number;  // 推理时间（毫秒）（仅在成功时存在）
  detections?: Detection[]; // 检测结果数组（仅在成功时存在）
  faceImagePaths?: string[]; // 裁切的人脸图片路径数组（仅在成功时存在）
  faceSimilarity?: number; // 人脸相似度（仅在成功时存在）
  isSamePerson?: boolean;  // 是否为同一人（仅在成功时存在）
  matchedReferencePath?: string; // 匹配的参考人脸图片路径（仅在识别为同一人时存在）
}

// 模型文件保存结果接口
interface SaveModelResult {
  success: boolean;        // 保存是否成功
  message: string;         // 保存结果消息
  savedModelPath?: string; // 保存的模型文件路径（仅在成功时存在）
}

// 人脸数据库图片保存结果接口
interface FaceDatabaseSaveResult {
  success: boolean;        // 保存是否成功
  message: string;         // 保存结果消息
  savedCount: number;      // 成功保存的图片数量
  totalAttempted: number;  // 尝试保存的图片总数
}

export const processImageWithRKNN: (photoUri: string, resMgr: resourceManager.ResourceManager, sandboxDir: string) => Promise<RknnProcessResult>;
export const saveModelFile: (resMgr: resourceManager.ResourceManager, sandboxDir: string) => SaveModelResult;
export const saveFaceDatabaseImages: (resMgr: resourceManager.ResourceManager, sandboxDir: string) => FaceDatabaseSaveResult;

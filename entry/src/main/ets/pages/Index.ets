import image from '@ohos.multimedia.image'
import CameraService from './CameraService';
import Logger from './Logger';
import fs from '@ohos.file.fs';
import { RknnProcessResult, Detection, ImageInfo, SaveModelResult, FaceDatabaseSaveResult } from '../../cpp/types/libentry/Index';
import prompt from '@ohos.promptAction';
import rknn from 'libentry.so';

// 定义一个接口来描述CameraMode的形状
interface CameraModeInterface {
  MODE_PHOTO: 0;
  MODE_VIDEO: 1;
}

// 使用该接口来定义CameraMode
const CameraMode: CameraModeInterface = {
  MODE_PHOTO: 0, // 拍照模式
  MODE_VIDEO: 1  // 录像模式
};

interface FileInfo {
  name: string;
  path: string;
  timestamp: number;
}


@Entry
@Component
struct Index {
  @State pipelineStartTime: number = 0;
  @State rotationAngle: number = 90;
  @State title: string = '人脸识别Demo'
  @State message: string = 'Demo说明：检查摄像头有无异常，无异常即可通过测试'
  private tag: string = 'FaceTest'
  private mXComponentController: XComponentController = new XComponentController()
  private surfaceId: string = '-1'
  private cameraService: CameraService = new CameraService(getContext(this))
  @State imgThumbnail: string = ''
  @State curModel: number = CameraMode.MODE_PHOTO
  // 添加显示沙箱图片的状态变量
  @State sandboxImagePath: string = ''
  @State showSandboxImage: boolean = false
  //=============real-time data start===============
  @State totalToggleTimesOfSettings: number = 10
  @State workedTotalToggleTimes: number = 0
  @State successToggleTimes: number = 0
  @State failToggleTimes: number = 0
  @State totalTakePhotoTimes: number = 0
  @State successTakePhotoTimes: number = 0
  @State failTakePhotoTimes: number = 0
  @State logAndPhotoSavePath: string = ''
  //============= real-time data end ===============

  //================switch start====================
  @State switchToggled: boolean = false
  @State switchTakePhotoToggled: boolean = false
  @State switchIntervalTime: number = 5     //间隔时间，单位秒
  //================ switch end ====================

  //================preview start=====================
  @State previewToggled: boolean = false
  @State previewTime: number = 72 // hour
  @State previewTakePhotoToggled: boolean = false
  @State previewTakePhotoIntervalTime: number = 5
  previewTextTimerController: TextTimerController = new TextTimerController()
  @State previewTextTimerFormat: string = 'hh:mm:ss'
  //================ preview end =====================

  //===============take photo start===================
  @State takePhotoToggled: boolean = false
  @State takePhotoTimesSettings: number = 50 // hour
  @State takePhotoIntervalTime: number = 5
  @State takePhotoNumberLimit: number = 500
  @State stopRunning: boolean = false
  //=============== take photo end ===================

  @State photomessage: string = 'Hello World';
  @State copyResult: string = '';
  @State resultImagePath: string = '';
  @State showResultImage: boolean = false;
  @State isImageEnlarged: boolean = false;
  // 添加检测结果状态变量
  @State detections: Detection[] = [];
  @State imageInfo: ImageInfo | null = null;
  @State faceImagePaths: string[] = []; // 裁切的人脸图片路径数组
  @State faceSimilarity: number = 0; // 人脸相似度
  @State isSamePerson: boolean = false; // 是否为同一人
  @State matchedReferencePath: string = ''; // 匹配的参考人脸路径
  @State testFaceImagePath: string = ''; // test_face_preprocessed.jpg的路径
  dir: string = getContext(this).filesDir + "/";
  // 自动拍照相关变量
  @State isAutoPhotoRunning: boolean = false;
  @State autoPhotoCount: number = 0;
  @State inferenceCounter: number = 0; // 推理计数器
  @State autoPhotoInterval: number = 10; // 自动拍照间隔，单位毫秒
  @State testResult: boolean = false;
  // 流水线处理相关状态
  @State frameCounter: number = 0; // 帧计数器
  @State isInferenceRunning: boolean = false; // 推理是否正在运行
  private readonly INFERENCE_INTERVAL: number = 1; // 每3张推理1张，提高人脸检测实时性

  // 流水线处理图片，实现每三张推理一张的机制
  private async processImagePipeline(photoUri: string): Promise<void> {
    this.frameCounter++;
    this.pipelineStartTime = Date.now();
    Logger.info(this.tag, `收到第${this.frameCounter}张图片: ${photoUri}`);

    // 判断是否需要推理
    const shouldInference = (this.frameCounter % this.INFERENCE_INTERVAL === 0);

    if (shouldInference && !this.isInferenceRunning) {
      // 需要推理且当前没有推理在运行
      Logger.info(this.tag, `第${this.frameCounter}张图片，开始RKNN推理`);
      this.isInferenceRunning = true;

      // 后台异步推理，不阻塞UI
      this.processImageWithInference(photoUri).then(() => {
        this.isInferenceRunning = false;
      }).catch((error: Error) => {
        Logger.error(this.tag, `推理处理失败: ${error.message}`);
        this.isInferenceRunning = false;
      });
    } else {
      // 不需要推理，直接保存为out.jpg并显示
      Logger.info(this.tag, `第${this.frameCounter}张图片，直接保存显示`);
      this.saveImageDirectly(photoUri);
    }
  }

  // 异步推理处理（后台运行）
  private async processImageWithInference(photoUri: string): Promise<void> {
    try {

      const result: RknnProcessResult =
        await rknn.processImageWithRKNN(photoUri, getContext(this).resourceManager, getContext(this).filesDir);
      if (result.success) {
        const totalCost = Date.now() - this.pipelineStartTime;
        Logger.info(this.tag, 'RKNN处理成功:', result.message);
        Logger.info(this.tag, '推理时间:', result.inferenceTime + 'ms');
        Logger.info(this.tag, '保存路径:', result.savedImagePath);

        // 保存检测结果和图像信息
        this.detections = result.detections || [];
        this.imageInfo = result.imageInfo || null;
        this.faceImagePaths = result.faceImagePaths || []; // 保存人脸图片路径
        this.faceSimilarity = result.faceSimilarity || 0; // 保存人脸相似度
        this.isSamePerson = result.isSamePerson || false; // 保存是否为同一人
        this.matchedReferencePath = result.matchedReferencePath || ''; // 保存匹配的参考人脸路径
        this.testFaceImagePath = getContext(this).filesDir + '/test_face_preprocessed.jpg'; // 设置test_face_preprocessed.jpg路径

        // 更新显示的图片为原始图片（不带框）
        this.imgThumbnail = 'file://' + result.savedImagePath;
        setTimeout(() => {
          let  totalCost = Date.now() - this.pipelineStartTime;
          Logger.info(this.tag, `真正显示完成耗时: ${totalCost}ms`);
        }, 0);
        Logger.info(this.tag, `保存图片成功: ${result.savedImagePath}`);
        Logger.info(this.tag, `检测到${this.detections.length}个人脸，更新框`);
        Logger.info(this.tag, `裁切了${this.faceImagePaths.length}张人脸图片`);
        Logger.info(this.tag, `人脸识别结果: 相似度=${this.faceSimilarity}, 是否同一人=${this.isSamePerson}`);
      } else {
        Logger.error(this.tag, 'RKNN处理失败:', result.message);
        // 清空检测结果
        this.detections = [];
        this.imageInfo = null;
        this.faceImagePaths = []; // 清空人脸图片路径
      }
    } catch (error) {
      Logger.error(this.tag, '调用RKNN处理接口失败:', error.message);
      // 清空检测结果
      this.detections = [];
      this.imageInfo = null;
      this.faceImagePaths = []; // 清空人脸图片路径
    }
  }

  // 直接保存图片为out.jpg格式并显示
  private async saveImageDirectly(photoUri: string): Promise<void> {
    try {
      // 获取沙箱路径
      const filesDir = getContext(this).filesDir;
      const timestamp = Date.now();
      const outputFileName = `out_${timestamp}.jpg`;
      const outputPath = `${filesDir}/${outputFileName}`;

      // 复制预览帧图片到out_*.jpg
      const sourceFile = photoUri.replace('file://', '');
      await fs.copyFile(sourceFile, outputPath);

      Logger.info(this.tag, `直接保存图片成功: ${outputPath}`);

      // 立即更新UI显示
      this.imgThumbnail = `file://${outputPath}`;


      // 管理out_*.jpg文件数量（保持最多20张）
      this.manageOutputFiles(filesDir);

    } catch (error) {
      Logger.error(this.tag, `直接保存图片失败: ${error}`);
    }
  }

  // 管理输出文件数量
  private async manageOutputFiles(filesDir: string): Promise<void> {
    try {
      const files = fs.listFileSync(filesDir);
      const outFiles: FileInfo[] = files.filter(file => file.startsWith('out_') && file.endsWith('.jpg'))
        .map(file => ({
          name: file,
          path: `${filesDir}/${file}`,
          timestamp: parseInt(file.match(/out_(\d+)\.jpg/)?.[1] || '0')
        } as FileInfo))
        .sort((a, b) => a.timestamp - b.timestamp);

      if (outFiles.length > 20) {
        const filesToDelete = outFiles.slice(0, 10);
        for (const file of filesToDelete) {
          fs.unlinkSync(file.path);
          Logger.info(this.tag, `删除旧的输出文件: ${file.name}`);
        }
      }
    } catch (error) {
      Logger.error(this.tag, `管理输出文件失败: ${error}`);
    }
  }

  async aboutToAppear() {
    Logger.info(this.tag, "CameraTestPage aboutToAppear")
    this.surfaceId = this.mXComponentController.getXComponentSurfaceId()
    Logger.info(this.tag, `aboutToAppear,surfaceId=${this.surfaceId}`)

    // 应用启动时自动保存模型文件
    try {
      Logger.info(this.tag, '应用启动时自动保存模型文件');
      const result: SaveModelResult = rknn.saveModelFile(getContext(this).resourceManager, getContext(this).filesDir);
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
      prompt.showToast({
        message: `调用保存模型文件接口失败: ${error}`,
        duration: 3000
      });
    }

    // 应用启动时自动保存人脸数据库图片
    try {
      Logger.info(this.tag, '应用启动时自动保存人脸数据库图片');
      const faceDbResult: FaceDatabaseSaveResult = rknn.saveFaceDatabaseImages(getContext(this).resourceManager, getContext(this).filesDir);
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
      prompt.showToast({
        message: `调用保存人脸数据库图片接口失败: ${error}`,
        duration: 3000
      });
    }

    // 设置拍照回调，用于更新UI显示图片
    this.cameraService.setTakePictureCallback((photoUri: string) => {
      Logger.info(this.tag, `Preview frame received: ${photoUri}`);
      // 使用流水线处理机制
      this.processImagePipeline(photoUri);
    })
    return
  }


  async initCamera(): Promise<boolean> {
    Logger.debug(this.tag, 'start initCamera')
    const isSuccess = await this.cameraService.initCamera(this.surfaceId)
    Logger.debug(this.tag, 'isSuccess  :' + isSuccess)
    this.testResult = isSuccess;
    return isSuccess
  }


  build() {
    Row() {
      // 左侧：摄像头预览区域
      Column() {
        Text('摄像头预览')
          .fontSize(20)
          .margin({ bottom: 10 })

        XComponent({
          id: 'componentId',
          type: 'surface',
          controller: this.mXComponentController
        }).onLoad(() => {
          if (!this.stopRunning) {
            Logger.info(this.tag, 'onLoad is called')
            this.mXComponentController.setXComponentSurfaceSize({ surfaceWidth: 640, surfaceHeight: 480 })
            this.surfaceId = this.mXComponentController.getXComponentSurfaceId()
            Logger.info(this.tag, `onLoad surfaceId: ${this.surfaceId}`)
            this.curModel = CameraMode.MODE_PHOTO
            this.initCamera()
          }
        })
          .width('80%')
          .height('70%')
          .rotate({
            angle: (this.rotationAngle - 90) % 360,
            centerX: '50%',
            centerY: '50%'
          })

        // 添加占位空间，使XComponent与右侧Image对齐
        Blank()
      }
      .width('50%')
      .height('100%')
      .justifyContent(FlexAlign.Start)
      .alignItems(HorizontalAlign.Center)
      .padding(10)
      //右侧：图片显示区域
      Column() {
        Text('拍摄图片')
          .fontSize(20)
          .margin({ bottom: 10 })

        // 显示处理后的图片和检测框
        if (this.imgThumbnail) {
          Stack({ alignContent: Alignment.TopStart }) {
            // 底层图片
            Image(this.imgThumbnail)
              .width('100%')
              .height('100%')
              .objectFit(ImageFit.Contain)
              .borderRadius(10)
              .backgroundColor('#f1f1f1')
              .onClick(() => {
                Logger.info(this.tag, `点击查看图片: ${this.imgThumbnail}`)
              })

            // 上层检测框（优化性能，避免频繁重绘）
            if (this.detections.length > 0 && this.imageInfo && this.imgThumbnail) {
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
                    .padding({ left: 3, right: 3, top: 1, bottom: 1 })
                    .borderRadius(2)
                    .position({ x: 0, y: -18 })
                }
                .position({
                  x: `${(detection.left / this.imageInfo!.width * 100)}%`,
                  y: `${(detection.top / this.imageInfo!.height * 100)}%`
                })
              }, (detection: Detection, index: number) => `${this.imgThumbnail}_detection_${index}`)
            }
          }
          .width('80%')
          .height('70%')
        } else {
          Text('拍照后将在此处显示沙箱图片')
            .width('100%')
            .height('70%')
            .textAlign(TextAlign.Center)
            .backgroundColor('#f1f1f1')
            .borderRadius(10)
            .fontSize(16)
        }

        Text('点击图片可查看详情')
          .fontSize(14)
          .fontColor('#666666')
          .margin({ top: 10 })

        // 显示匹配的参考人脸（如果识别为同一人）
        if (this.isSamePerson && this.matchedReferencePath) {
          Text('匹配的参考人脸')
            .fontSize(16)
            .fontWeight(FontWeight.Bold)
            .margin({ top: 15, bottom: 10 })

          // 显示匹配的参考人脸图片
          Column() {
            Image('file://' + this.matchedReferencePath)
              .width(120)
              .height(120)
              .objectFit(ImageFit.Cover)
              .borderRadius(8)
              .border({ width: 2, color: Color.Green })
              .onClick(() => {
                Logger.info(this.tag, `点击查看匹配的参考人脸: ${this.matchedReferencePath}`);
              })
            
            Text('匹配人脸')
              .fontSize(12)
              .fontColor('#666666')
              .margin({ top: 5 })
          }
          .margin({ bottom: 15 })
        }

        // 显示模型文件状态的文本，替代原来的保存按钮
        Text('模型文件已在应用启动时自动保存')
          .fontSize(14)
          .fontColor('#007DFF')
          .margin({ top: 10 })
      }
      .width('50%')
      .height('100%')
      .justifyContent(FlexAlign.Start)
      .alignItems(HorizontalAlign.Center)
      .padding(10)
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.White)
    }
}

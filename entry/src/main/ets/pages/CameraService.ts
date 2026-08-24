/*
 * Copyright (c) 2022 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import camera from '@ohos.multimedia.camera'
import deviceInfo from '@ohos.deviceInfo'
import fileio from '@ohos.fileio'
import image from '@ohos.multimedia.image'
import media from '@ohos.multimedia.media'
import Logger from './Logger'
import MediaUtils from './MediaUtils'
import { photoAccessHelper } from '@kit.MediaLibraryKit';
import { BusinessError } from '@kit.BasicServicesKit'
import { Context } from '@kit.AbilityKit'
import fs from '@ohos.file.fs'
import { common } from '@kit.AbilityKit'


const CameraMode = {
  MODE_PHOTO: 0, // 拍照模式
  MODE_VIDEO: 1 // 录像模式
}

const CameraSize = {
  WIDTH: 640,
  HEIGHT: 480
}

export default class CameraService {
  private tag: string = 'CameraService'
  private context: any = undefined
  private mediaUtil: MediaUtils = undefined
  private cameraManager: camera.CameraManager = undefined
  private cameras: Array<camera.CameraDevice> = undefined
  private cameraId: string = ''
  private cameraInput: camera.CameraInput = undefined
  private previewOutput: camera.PreviewOutput = undefined
  private photoOutPut: camera.PhotoOutput = undefined
  private captureSession: camera.CaptureSession = undefined
  private mReceiver: image.ImageReceiver = undefined
  private photoUri: string = ''
  private fileAsset: photoAccessHelper.PhotoAsset = undefined
  private fd: number = -1
  private curMode = CameraMode.MODE_PHOTO
  // private videoRecorder: media.VideoRecorder = undefined
  private videoRecorder: media.AVRecorder = undefined;
  private videoOutput: camera.VideoOutput = undefined
  private handleTakePicture: (photoUri: string) => void = undefined
  private cameraOutputCapability: camera.CameraOutputCapability = undefined
  private autoCaptureTimer: number = -1
  private photoCaptureInProgress: boolean = false
  private videoConfig: any = {
    audioSourceType: 1,
    videoSourceType: 0,
    profile: {
      audioBitrate: 48000,
      audioChannels: 2,
      audioCodec: 'audio/mp4v-es',
      audioSampleRate: 48000,
      durationTime: 1000,
      fileFormat: 'mp4',
      videoBitrate: 48000,
      videoCodec: 'video/mp4v-es',
      videoFrameWidth: 640,
      videoFrameHeight: 480,
      videoFrameRate: 30
    },
    url: '',
    orientationHint: 0,
    location: {
      latitude: 30, longitude: 130
    },
    maxSize: 10000,
    maxDuration: 10000
  }

  constructor(context: any) {
    this.context = context
    this.mediaUtil = MediaUtils.getInstance(context)
    // 创建ImageReceiver用于获取预览流图片
    this.mReceiver = image.createImageReceiver(CameraSize.WIDTH, CameraSize.HEIGHT, image.ImageFormat.JPEG, 8)
    Logger.debug(this.tag, 'createImageReceiver for preview stream')
    
    // 设置预览流图片到达回调
    this.mReceiver.on('imageArrival', () => {
      Logger.debug(this.tag, 'Preview frame imageArrival')
      this.mReceiver.readNextImage((err: BusinessError, nextImage: image.Image) => {
        Logger.debug(this.tag, 'readNextImage from preview stream')
        if (err || nextImage === undefined) {
          Logger.error(this.tag, 'failed to get valid preview image')
          return
        }
        nextImage.getComponent(image.ComponentType.JPEG, (err: BusinessError, imgComponent: image.Component) => {
          Logger.debug(this.tag, 'getComponent from preview stream')
          if (err || imgComponent === undefined) {
            Logger.debug(this.tag, 'failed to get valid buffer from preview')
            return
          }
          let buffer = new ArrayBuffer(4096)
          if (imgComponent.byteBuffer) {
            buffer = imgComponent.byteBuffer
            Logger.debug(this.tag,'Preview stream buffer size:' + buffer.byteLength)
          } else {
            Logger.error(this.tag, 'Preview stream byteBuffer is undefined')
          }
          // 保存预览流图片并回调到UI
          this.savePreviewPicture(buffer, nextImage)
        })
      })
    })
  }

  // 保存预览流图片的方法
  async savePreviewPicture(buffer: ArrayBuffer, img: image.Image) {
    Logger.debug(this.tag, 'savePreviewPicture from preview stream');
    try {
      // 获取应用沙箱路径
      const context = this.context as common.UIAbilityContext;
      const filesDir = context.filesDir;
      // 使用时间戳生成唯一文件名
      const timestamp = Date.now();
      const uniqueFileName = `preview_frame_${timestamp}.jpg`;
      const filePath = `${filesDir}/${uniqueFileName}`;
      
      Logger.info(this.tag, `Saving preview frame to sandbox path: ${filePath}`);

      // 使用OpenHarmony图片处理API将YUV数据转换为JPEG
       let opts: image.InitializationOptions = { editable: true, pixelFormat: image.PixelMapFormat.RGBA_8888, size: { height: 480, width: 640 } }
       const pixelMap = await image.createPixelMap(buffer,opts);
       const imageInfo = await pixelMap.getImageInfo();
       Logger.info(this.tag, `Created PixelMap from YUV data, size: ${imageInfo.size.width}x${imageInfo.size.height}`);

       // 修正颜色通道顺序 - 交换红蓝通道
       const pixelBuffer = new ArrayBuffer(imageInfo.size.width * imageInfo.size.height * 4);
       await pixelMap.readPixelsToBuffer(pixelBuffer);
       const pixels = new Uint8Array(pixelBuffer);

       // 创建修正后的PixelMap
       const correctedOpts: image.InitializationOptions = { editable: true, pixelFormat: image.PixelMapFormat.BGRA_8888, size: { height: 480, width: 640 } }
       const correctedPixelMap = await image.createPixelMap(pixelBuffer, correctedOpts);
       
       const imagePackerApi = image.createImagePacker();
       const packOpts: image.PackingOption = {
         format: 'image/jpeg',
         quality: 98
       }
       const jpegBuffer = await imagePackerApi.packing(correctedPixelMap, packOpts);
       Logger.info(this.tag, `Converted YUV to JPEG with color correction, buffer size: ${jpegBuffer.byteLength}`);
       
       // 释放资源
       await pixelMap.release();
       await correctedPixelMap.release();
       await imagePackerApi.release();
      
      // 创建文件并写入JPEG数据
      const file = fs.openSync(filePath, fs.OpenMode.CREATE | fs.OpenMode.WRITE_ONLY);
      fs.writeSync(file.fd, jpegBuffer);
      fs.closeSync(file);
      
      this.photoUri = `file://${filePath}`;
      
      // 将保存的图片路径输出到hilog
      Logger.info(this.tag, `Preview frame saved successfully as JPEG to sandbox, URI: ${this.photoUri}`);
      
      // 释放图像资源
      await img.release();
      
      // 调用回调函数，将预览帧传递给UI
      if (this.handleTakePicture) {
        this.handleTakePicture(this.photoUri);
      }
    } catch (error) {
      Logger.error(this.tag, `Failed to save preview frame to sandbox: ${error}`);
    }
  }

  // 原来的拍照保存方法（保留用于手动拍照）
  async savePicture(buffer: ArrayBuffer, img: image.Image) {
    Logger.debug(this.tag, 'savePicture using sandbox path');
    try {
      // 获取应用沙箱路径
      const context = this.context as common.UIAbilityContext;
      const filesDir = context.filesDir;
      // 使用时间戳生成唯一文件名
      const timestamp = Date.now();
      const uniqueFileName = `camera_photo_${timestamp}.jpg`;
      const filePath = `${filesDir}/${uniqueFileName}`;
      
      Logger.info(this.tag, `Saving photo to sandbox path: ${filePath}`);
      
      // 删除之前的照片文件（保持只有一张最新照片）
      try {
        const files = fs.listFileSync(filesDir);
        for (const file of files) {
          if (file.startsWith('camera_photo_') && file.endsWith('.jpg')) {
            const oldFilePath = `${filesDir}/${file}`;
            fs.unlinkSync(oldFilePath);
            Logger.info(this.tag, `Deleted old photo: ${oldFilePath}`);
          }
        }
      } catch (error) {
        Logger.info(this.tag, 'No previous photo files to delete');
      }
      
      // 创建文件并写入数据
      const file = fs.openSync(filePath, fs.OpenMode.CREATE | fs.OpenMode.WRITE_ONLY);
      fs.writeSync(file.fd, buffer);
      fs.closeSync(file);
      
      this.photoUri = `file://${filePath}`;
      
      // 将保存的图片路径输出到hilog
      Logger.info(this.tag, `Photo saved successfully to sandbox, URI: ${this.photoUri}`);
      
      // 调用回调函数
      if (this.handleTakePicture) {
        this.handleTakePicture(this.photoUri);
      }
    } catch (error) {
      Logger.error(this.tag, `Failed to save picture to sandbox: ${error}`);
    } finally {
      await img.release();
      this.photoCaptureInProgress = false;
    }
  }



  async initCamera(surfaceId: string, serviceRetryCount: number = 0): Promise<boolean> {
    try {
      console.info('[USB_PREVIEW] initCamera start')
      await this.releaseCamera()
      Logger.debug(this.tag, `deviceInfo.deviceType = ${deviceInfo.deviceType}`)
      if (deviceInfo.deviceType === 'default') {
        this.videoConfig.videoSourceType = 1
      } else {
        this.videoConfig.videoSourceType = 0
      }
      try {
        this.cameraManager = await camera.getCameraManager(this.context)
        console.info('[USB_PREVIEW] getCameraManager success')
      } catch (error) {
        let err = error as BusinessError;
        console.error(`[USB_PREVIEW] getCameraManager failed, code=${err.code}, message=${err.message}`)
        return false
      }
      this.cameras = await this.cameraManager.getSupportedCameras()
      console.info(`[USB_PREVIEW] supported camera count=${this.cameras.length}`)
      if (this.cameras.length === 0) {
        console.error('[USB_PREVIEW] no camera is exposed by Camera Kit')
        return false
      }

      this.cameras.forEach((device: camera.CameraDevice, index: number) => {
        console.info(`[USB_PREVIEW] camera[${index}]=${JSON.stringify(device)}`)
      })

      let cameraDevice: camera.CameraDevice | undefined = this.cameras.find((device: camera.CameraDevice) => {
        return device.connectionType === camera.ConnectionType.CAMERA_CONNECTION_USB_PLUGIN
      })
      if (cameraDevice === undefined) {
        console.error('[USB_PREVIEW] Camera Kit did not expose a USB camera (connectionType=1)')
        return false
      }

      console.info(`[USB_PREVIEW] selected USB camera=${JSON.stringify(cameraDevice)}`)
      this.cameraInput = await this.cameraManager.createCameraInput(cameraDevice)
      await this.cameraInput.open()
      console.info('[USB_PREVIEW] cameraInput opened')
      this.cameraOutputCapability = await this.cameraManager.getSupportedOutputCapability(cameraDevice)
      console.info(`[USB_PREVIEW] previewProfiles=${JSON.stringify(this.cameraOutputCapability.previewProfiles)}`)
      console.info(`[USB_CAPTURE] photoProfiles=${JSON.stringify(this.cameraOutputCapability.photoProfiles)}`)
      if (this.cameraOutputCapability.previewProfiles.length === 0) {
        console.error('[USB_PREVIEW] selected USB camera has no preview profile')
        return false
      }

      let previewProfile: camera.Profile | undefined = this.cameraOutputCapability.previewProfiles.find(
        (profile: camera.Profile) => {
          return profile.size.width === CameraSize.WIDTH && profile.size.height === CameraSize.HEIGHT
        }
      )
      if (previewProfile === undefined) {
        previewProfile = this.cameraOutputCapability.previewProfiles[0]
        console.info(`[USB_PREVIEW] 640x480 profile unavailable, fallback=${JSON.stringify(previewProfile)}`)
      } else {
        console.info(`[USB_PREVIEW] selected 640x480 profile=${JSON.stringify(previewProfile)}`)
      }

      this.previewOutput = await this.cameraManager.createPreviewOutput(previewProfile, surfaceId)
      this.previewOutput.on('frameStart', () => {
        console.info('[USB_PREVIEW] frameStart')
      })
      this.previewOutput.on('frameEnd', () => {
        console.info('[USB_PREVIEW] frameEnd')
      })
      this.previewOutput.on('error', (previewOutputError) => {
        console.error(`[USB_PREVIEW] previewOutput error code=${previewOutputError.code}`)
      })

      console.info('[USB_PREVIEW] previewOutput created')

      let photoProfile: camera.Profile | undefined = this.cameraOutputCapability.photoProfiles.find(
        (profile: camera.Profile) => {
          return profile.size.width === CameraSize.WIDTH && profile.size.height === CameraSize.HEIGHT
        }
      )
      if (photoProfile === undefined && this.cameraOutputCapability.photoProfiles.length > 0) {
        photoProfile = this.cameraOutputCapability.photoProfiles[0]
      }
      if (photoProfile !== undefined) {
        this.photoOutPut = this.cameraManager.createPhotoOutput(photoProfile)
        this.setPhotoOutputCb(this.photoOutPut, this.context as Context)
        console.info(`[USB_CAPTURE] photoOutput created, profile=${JSON.stringify(photoProfile)}`)
      } else {
        console.error('[USB_CAPTURE] USB camera exposes no photo profile')
      }

      // UI 预览保持单路 Surface；右侧图片使用正规的 PhotoOutput 获取 JPEG，
      // 不再把 JPEG ImageReceiver 错误绑定到 RGBA PreviewProfile。
      this.captureSession = await this.cameraManager.createCaptureSession()
      console.info('[USB_PREVIEW] captureSession created')
      await this.captureSession.beginConfig()
      await this.captureSession.addInput(this.cameraInput)
      await this.captureSession.addOutput(this.previewOutput)
      if (this.photoOutPut !== undefined) {
        await this.captureSession.addOutput(this.photoOutPut)
      }
      await this.captureSession.commitConfig()
      console.info('[USB_PREVIEW] captureSession committed')
      await this.captureSession.start()
      console.info('[USB_PREVIEW] captureSession started')
      if (this.photoOutPut !== undefined) {
        this.startAutoCapture()
      }
    } catch (error) {
      let err = error as BusinessError
      console.error(`[USB_PREVIEW] init failed, code=${err.code}, message=${err.message}`)
      // 该板卡在 USB 摄像头首次打开时可能重启 camera_service，并返回 7400201。
      // 服务恢复后重新获取 CameraManager/CameraDevice 即可正常打开，最多自动重试一次。
      if (err.code === 7400201 && serviceRetryCount < 1) {
        console.info('[USB_PREVIEW] camera service restarted, retrying once after 1500 ms')
        await new Promise<void>((resolve: () => void) => {
          setTimeout(resolve, 1500)
        })
        return this.initCamera(surfaceId, serviceRetryCount + 1)
      }
      return false
    }
    return true
  }

  setTakePictureCallback(callback) {
    this.handleTakePicture = callback
  }

  setPhotoOutputCb(photoOutput: camera.PhotoOutput, context: Context) {
    //设置回调之后，调用photoOutput的capture方法，就会将拍照的buffer回传到回调中。
    photoOutput.on('photoAvailable', (errCode: BusinessError, photo: camera.Photo): void => {
      Logger.debug(this.tag, 'getPhoto start');
      Logger.debug(this.tag, `err: ${errCode}`);
      if (errCode || photo === undefined) {
        Logger.error(this.tag, 'getPhoto failed');
        this.photoCaptureInProgress = false;
        return;
      }
      let imageObj: image.Image = photo.main;
      imageObj.getComponent(image.ComponentType.JPEG, (errCode: BusinessError, component: image.Component): void => {
        Logger.debug(this.tag, 'getComponent start');
        if (errCode || component === undefined) {
          Logger.error(this.tag, 'getComponent failed');
          imageObj.release();
          this.photoCaptureInProgress = false;
          return;
        }
        let buffer: ArrayBuffer;
        if (component.byteBuffer) {
          buffer = component.byteBuffer;
          Logger.debug(this.tag,"buffer:"+buffer.byteLength);
        } else {
          Logger.error(this.tag,'byteBuffer is null');
          imageObj.release();
          this.photoCaptureInProgress = false;
          return;
        }
        // 如需要在图库中看到所保存的图片、视频资源，请使用用户无感的安全控件创建媒体资源。
        this.savePicture(buffer, imageObj);
      });
    });
  }

  private startAutoCapture(): void {
    this.stopAutoCapture()
    console.info('[USB_CAPTURE] auto capture started, interval=1500ms')
    this.autoCaptureTimer = setInterval(() => {
      if (this.photoOutPut === undefined || this.photoCaptureInProgress) {
        return
      }
      this.photoCaptureInProgress = true
      this.photoOutPut.capture().then(() => {
        console.info('[USB_CAPTURE] capture request submitted')
      }).catch((error: BusinessError) => {
        console.error(`[USB_CAPTURE] capture failed, code=${error.code}, message=${error.message}`)
        this.photoCaptureInProgress = false
      })
    }, 1500)
  }

  private stopAutoCapture(): void {
    if (this.autoCaptureTimer !== -1) {
      clearInterval(this.autoCaptureTimer)
      this.autoCaptureTimer = -1
    }
    this.photoCaptureInProgress = false
  }

  async takePicture() {
    Logger.debug(this.tag, 'takePicture')
    try {
      if (this.curMode === CameraMode.MODE_VIDEO) {
        this.curMode = CameraMode.MODE_PHOTO
      }
      let photoSettings = {
        rotation: camera.ImageRotation.ROTATION_0,
        quality: camera.QualityLevel.QUALITY_LEVEL_MEDIUM,
        location: {
          // 位置信息，经纬度
          latitude: 12.9698,
          longitude: 77.7500,
          altitude: 1000
        },
        mirror: false
      }
      await this.photoOutPut.capture(photoSettings)
      Logger.debug(this.tag, 'takePicture done')
      // 照片buffer会通过photoAvailable回调获取，不需要在这里处理
      AppStorage.Set('isRefresh', true)
    }catch (error) {
      Logger.error(this.tag, `拍照失败: ${error}`)
    }
  }

  async startVideo() {
    Logger.debug(this.tag, 'startVideo begin')
    await this.captureSession.stop()
    await this.captureSession.beginConfig()
    if (this.curMode === CameraMode.MODE_PHOTO) {
      this.curMode = CameraMode.MODE_VIDEO
      if (this.photoOutPut) {
        await this.captureSession.removeOutput(this.photoOutPut)
        this.photoOutPut.release()
      }
    } else {
      if (this.videoOutput) {
        await this.captureSession.removeOutput(this.videoOutput)
      }
    }
    if (this.videoOutput) {
      await this.captureSession.removeOutput(this.videoOutput)
      await this.videoOutput.release()
    }
    this.fileAsset = await this.mediaUtil.createAndGetUri(photoAccessHelper.PhotoType.VIDEO)
    this.fd = await this.mediaUtil.getFdPath(this.fileAsset)
    this.videoRecorder = await media.createAVRecorder()
    this.videoConfig.url = `fd://${this.fd}`
    await this.videoRecorder.prepare(this.videoConfig)
    let videoId = await this.videoRecorder.getInputSurface()
    let videoProfile = this.cameraOutputCapability.videoProfiles[0];
    this.videoOutput = await this.cameraManager.createVideoOutput(videoProfile, videoId)
    await this.captureSession.addOutput(this.videoOutput)
    await this.captureSession.commitConfig()
    await this.captureSession.start()
    await this.videoOutput.start()
    await this.videoRecorder.start()
    Logger.debug(this.tag, 'startVideo end')
  }

  async stopVideo() {
    Logger.debug(this.tag, 'stopVideo called')
    await this.videoRecorder.stop()
    await this.videoOutput.stop()
    await this.videoRecorder.release()
    await this.fileAsset.close(this.fd)
  }

  async releaseCamera() {
    Logger.debug(this.tag, 'releaseCamera')
    this.stopAutoCapture()
    if (this.cameraInput) {
      await this.cameraInput.close()
    }
    if (this.previewOutput) {
      await this.previewOutput.release()
    }
    if (this.photoOutPut) {
      await this.photoOutPut.release()
    }
    if (this.videoOutput) {
      await this.videoOutput.release()
    }
    if (this.captureSession) {
      await this.captureSession.release()
    }
  }
}

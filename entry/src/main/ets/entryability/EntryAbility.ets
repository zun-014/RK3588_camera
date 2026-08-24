import hilog from '@ohos.hilog';
import prompt from '@ohos.prompt';
import { AbilityConstant, ConfigurationConstant, Want , abilityAccessCtrl , Permissions , common , UIAbility } from '@kit.AbilityKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { window } from '@kit.ArkUI';
const PERMISSIONS: Array<Permissions> = ['ohos.permission.READ_MEDIA','ohos.permission.CAMERA','ohos.permission.WRITE_MEDIA']
const DOMAIN = 0x0000;

function reqPermissionsFromUser(permissions: Array<Permissions>, context: common.UIAbilityContext): void {
  let atManager: abilityAccessCtrl.AtManager = abilityAccessCtrl.createAtManager();
  // requestPermissionsFromUser会判断权限的授权状态来决定是否唤起弹窗。
  atManager.requestPermissionsFromUser(context, permissions).then((data) => {
    let grantStatus: Array<number> = data.authResults;
    let length: number = grantStatus.length;
    for (let i = 0; i < length; i++) {
      if (grantStatus[i] === 0) {
        // 用户授权，可以继续访问目标操作。
      } else {
        // 用户拒绝授权，提示用户必须授权才能访问当前页面的功能，并引导用户到系统设置中打开相应的权限。
        return;
      }
    }
    // 授权成功。
  }).catch((err: BusinessError) => {
    console.error(`Failed to request permissions from user. Code is ${err.code}, message is ${err.message}`);
  })
}

export default class EntryAbility extends UIAbility {

  onWindowStageCreate(windowStage: window.WindowStage): void {
    // ...
    windowStage.loadContent('pages/Index', (err, data) => {
      reqPermissionsFromUser(PERMISSIONS, this.context);
      // ...
    });
  }

  onDestroy(): void {
    hilog.info(DOMAIN, 'testTag', '%{public}s', 'Ability onDestroy');
  }



  onWindowStageDestroy(): void {
    // Main window is destroyed, release UI related resources
    hilog.info(DOMAIN, 'testTag', '%{public}s', 'Ability onWindowStageDestroy');
  }

  onForeground(): void {
    // Ability has brought to foreground
    hilog.info(DOMAIN, 'testTag', '%{public}s', 'Ability onForeground');
  }

  onBackground(): void {
    // Ability has back to background
    hilog.info(DOMAIN, 'testTag', '%{public}s', 'Ability onBackground');
  }
}
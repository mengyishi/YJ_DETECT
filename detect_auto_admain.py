import sys, os, win32api
import mss
import numpy as np
import cv2
import pydirectinput
import pynput
import time
import win32con
import win32gui
from bokeh.events import PressUp
from ultralytics.utils.plotting import Annotator, colors
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, increment_path, non_max_suppression, scale_boxes, xyxy2xywh
from utils.loggers.comet import CONF_THRES, IOU_THRES
from utils.torch_utils import select_device
import torch
import cv2
import numpy as np

#键盘控制
keyboard_controller = pynput.keyboard.Controller()

sct = mss.mss()
screen_width = 1920   #屏幕宽
screen_height = 1080  #屏幕高
center_width = 800  # 中间截图区域的宽度
center_height = 600  # 中间截图区域的高度
GAME_LEFT, GAME_TOP,GAME_WIDTH, GAME_HEIGHT = screen_width,screen_height ,screen_width ,screen_height  #游戏内截图区域
RESIZE_WIN_WIDTH, RESIZE_WIN_HEIGHT = screen_width , screen_height   #显示窗口大小
monitor = {
    'left': 300,   # 红框左上角的x坐标
    'top': 100,    # 红框左上角的y坐标
    'width': 400,  # 红框的宽度
    'height': 400  # 红框的高度
}
window_name = 'test'
WEIGHTS ='best.pt'
IMGSZ = [640,640]
CONF_THRES = 0.25
IOU_THRES = 0.1
LINE_THICKNESS = 1
HIDE_CONF = True
HIDE_LABELS = True
MAX_DET = 1000

def get_model():
    device = select_device('')
    half = device.type != 'cpu'
    model = attempt_load(WEIGHTS, device)

    stride = int(model.stride.max())
    names = model.module.names if hasattr(model, 'module') else model.names
    if half:
        model.half()
    return model, device, half, stride, names

model, device, half, stride, names = get_model()
imgsz = check_img_size(IMGSZ, s=stride)

#图像识别
@torch.no_grad()
def pred_img(img0):
    # 选GPU OR CPU
    img = letterbox(img0, IMGSZ, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()

    img = img / 255.0
    if len(img.shape) == 3:
        img = img[None]

    # 模型预测
    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, None, False, max_det=MAX_DET)

    _, det = next(enumerate(pred))
    im0 = img0.copy()
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
    annotator = Annotator(im0, line_width=LINE_THICKNESS, example=str(names))

    xywh_list = []
    detections_info = []  # 新增用于存储坐标和类别的列表

    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # 类别索引
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            xywh_list.append(xywh)

            # 获取类别名称和坐标，并添加到 detections_info
            label = names[c] if HIDE_LABELS else (f"{names[c]} {conf:.2f}" if not HIDE_CONF else None)
            detection_info = {
                'class_name': names[c],  # 类别名称
                'xyxy': [int(coord.item()) for coord in xyxy],  # 检测框的xyxy坐标
                'confidence': float(conf)  # 置信度
            }
            detections_info.append(detection_info)  # 将该检测结果添加到列表中

            # 可视化检测框和标签
            annotator.box_label(xyxy, label, color=colors(c, True))

    im0 = annotator.result()

    # 返回识别后的图像、xywh坐标列表，以及包含坐标和类别名称的列表
    return im0, xywh_list, detections_info

def press_key_control(detections_info):
    for detection in detections_info:
        class_name = detection['class_name']

        if len(class_name) == 1 and class_name.isalpha():
            print(f"Pressing key: {class_name}")
            pydirectinput.keyDown(class_name)
            time.sleep(0.05)
            pydirectinput.keyUp(class_name)
            #keyboard_controller.press(class_name)
            #keyboard_controller.release(class_name)
            time.sleep(0.1)

def is_admin():
    # 由于win32api中没有IsUserAnAdmin函数,所以用了这种方法
    try:
        # 在c:\windows目录下新建一个文件test01.txt
        testfile = os.path.join(os.getenv("windir"), "test01.txt")
        open(testfile, "w").close()
    except OSError:  # 不成功
        return False
    else:  # 成功
        os.remove(testfile)  # 删除文件
        return True


print(is_admin())
if is_admin():
    while True:
        img = sct.grab(monitor)
        img = np.array(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        black_background = np.zeros((1080, 1920, 3), dtype=np.uint8)
        start_x = (1920 - 400) // 2  # 中心位置的x坐标
        start_y = (1080 - 400) // 2  # 中心位置的y坐标
        black_background[start_y:start_y + 400, start_x:start_x + 400] = img
        img, aims, detect_name = pred_img(black_background)
        if detect_name:
            press_key_control(detect_name)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1920, 1080)  # 确保显示窗口也是1080p
        # cv2.resizeWindow(window_name, RESIZE_WIN_WIDTH, RESIZE_WIN_HEIGHT)
        cv2.imshow(window_name, img)

        k = cv2.waitKey(1)

        # 置顶
        hwnd = win32gui.FindWindow(None, window_name)

        win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)

        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                              win32con.SWP_NOMOVE | win32con.SWP_NOACTIVATE | win32con.SWP_NOOWNERZORDER | win32con.SWP_SHOWWINDOW |
                              win32con.SWP_NOSIZE
                              )

        if k % 256 == 27:  # esc
            cv2.destroyAllWindows()
            exit('结束img_show')

# 主程序写在这里

else:
    # 以管理员权限重新运行程序
    win32api.ShellExecute(None, "runas", sys.executable, __file__, None, 1)

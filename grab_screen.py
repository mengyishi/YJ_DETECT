import mss
import numpy as np
import cv2
import pynput
import win32con
import win32gui
from ultralytics.utils.plotting import Annotator, colors
from models.experimental import attempt_load
from mouse import move_mouse
from utils.augmentations import letterbox
from utils.general import check_img_size, increment_path, non_max_suppression, scale_boxes, xyxy2xywh
from utils.loggers.comet import CONF_THRES, IOU_THRES
from utils.torch_utils import select_device
import torch
import cv2
import numpy as np
#此为鼠标的   乱写，用不了
sct = mss.mss()
screen_width = 1920   #屏幕宽
screen_height = 1080  #屏幕高
center_width = 800  # 中间截图区域的宽度
center_height = 600  # 中间截图区域的高度

#加载鼠标控制
mouse_controller = pynput.mouse.Controller()


GAME_LEFT, GAME_TOP,GAME_WIDTH, GAME_HEIGHT = 300, 100, 400, 400  #游戏内截图区域
RESIZE_WIN_WIDTH, RESIZE_WIN_HEIGHT = screen_width , screen_height   #显示窗口大小
monitor = {
    'left': GAME_LEFT,   # 红框左上角的x坐标
    'top': GAME_TOP,    # 红框左上角的y坐标
    'width': GAME_WIDTH,  # 红框的宽度
    'height': GAME_HEIGHT  # 红框的高度
}
window_name = 'detect'
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
#选GPU OR CPU
    img = letterbox(img0, IMGSZ, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    #归一化处理
    img = img / 255.0
    if len(img.shape) == 3:
        img = img[None]
    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, None, False, max_det=MAX_DET)

    #_, det = next(enumerate(pred))
    det = pred[0]
    im0= img0.copy()
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
    annotator = Annotator(im0, line_width=LINE_THICKNESS, example=str(names))
    xywh_list = []
    if len(det):
            # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            xywh_list.append(xywh)
            #label = None if HIDE_LABELS else (names[c] if HIDE_CONF else f"{names[c]} {conf:.2f}")

            label = names[c] if HIDE_LABELS else (f"{names[c]} {conf:.2f}" if not HIDE_CONF else None)

            annotator.box_label(xyxy, label, color=colors(c, True))
    im0 = annotator.result()

    return im0, xywh_list

def mouse_aim_controller(xywh_list, mouse, left, top, width, height):
    #获取鼠标相对于屏幕的XY坐标
    mouse_x, mouse_y = mouse.position
    #获取监测区域的大小以及位置
    best_xy = None
    for xywh in xywh_list:
        x, y, _, _ = xywh
        x *= width
        y *= height

        x += left
        y += top

        dist = ((x - mouse_x) ** 2 + (y - mouse_y) ** 2) ** .5
        if not best_xy:
            best_xy = ((x,y),dist)
        else:
            _, old_dist = best_xy
            if dist < old_dist:
                best_xy = ((x,y),dist)
    #还原相对于监测区域的位置
    x,y = best_xy[0]

    sub_x, sub_y = x-mouse_x, y-mouse_y
    move_mouse(sub_x, sub_y)



while True:
    img = sct.grab(monitor)
    img = np.array(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    black_background = np.zeros((1080, 1920, 3), dtype=np.uint8)
    start_x = (1920 - 400) // 2  # 中心位置的x坐标
    start_y = (1080 - 400) // 2  # 中心位置的y坐标
    black_background[start_y:start_y + 400, start_x:start_x + 400] = img
    img, aims = pred_img(black_background)
    if aims:
        mouse_aim_controller(aims, mouse_controller,**monitor)

    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 1080)  # 确保显示窗口也是1080p
    #cv2.resizeWindow(window_name, RESIZE_WIN_WIDTH, RESIZE_WIN_HEIGHT)
    cv2.imshow(window_name, img  )

    k = cv2.waitKey(1)

    # 置顶
    hwnd = win32gui.FindWindow(None, window_name)

    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)

    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOACTIVATE | win32con.SWP_NOOWNERZORDER | win32con.SWP_SHOWWINDOW |
                          win32con.SWP_NOSIZE
                          )

    if k % 256 ==27:    #esc
        cv2.destroyAllWindows()
        exit('结束img_show')

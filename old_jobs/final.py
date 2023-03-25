import argparse
import asyncio
import base64
import configparser
import time
from pathlib import Path

import cv2
import imutils
import mediapipe as mp
import numpy as np
import torch
import websockets
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, \
    scale_coords, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


# import pyttsx3

class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        self.path = path
        self.img_size = img_size
        self.stride = stride

        path = path[22:]

        img_data = base64.b64decode(path)

        # with open('./data/images/base64.jpg', 'wb') as file:
        #     file.write(img_data)

        # print(img_data)
        img_array = np.fromstring(img_data, np.uint8)
        # self.img0 = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
        self.img0 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Padded resize
        img = letterbox(self.img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        self.img = np.ascontiguousarray(img)


async def detect(base64Data, websocket):
    dataset = LoadImages(base64Data, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    img = dataset.img
    im0s = dataset.img0

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]
    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()
    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s, im0, frame = '', im0s, getattr(dataset, 'frame', 0)
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_img or view_img:  # Add bbox to image
                    label = f'{names[int(cls)]}'
                    # if label == 'calling':
                    #     speaker.Speak('打电话')
                    # elif label == 'smoking':
                    #     speaker.Speak('抽烟')
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')
        global EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, MAR_THRESH, MOUTH_AR_CONSEC_FRAMES, COUNTER, TOTAL, mCOUNTER, mTOTAL
        # Stream results
        im0 = imutils.resize(im0, width=720)
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        results = model2.process(im0)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=im0,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
            h, w = im0.shape[0], im0.shape[1]
            Lei = results.multi_face_landmarks[0].landmark[133]  # 左眼内眼角
            Lei_X, Lei_Y = int(Lei.x * w), int(Lei.y * h);
            Leo = results.multi_face_landmarks[0].landmark[33]  # 左眼外眼角
            Leo_X, Leo_Y = int(Leo.x * w), int(Leo.y * h);
            Letl = results.multi_face_landmarks[0].landmark[160];  # 左眼上左
            Letl_X, Letl_Y = int(Letl.x * w), int(Letl.y * h);
            Letr = results.multi_face_landmarks[0].landmark[158];  # 左眼上右
            Letr_X, Letr_Y = int(Letr.x * w), int(Letl.y * h);
            Lebl = results.multi_face_landmarks[0].landmark[144];  # 左眼下左
            Lebl_X, Lebl_Y = int(Lebl.x * w), int(Lebl.y * h);
            Lebr = results.multi_face_landmarks[0].landmark[153];  # 左眼下右
            Lebr_X, Lebr_Y = int(Lebr.x * w), int(Lebl.y * h);
            Leftear = (Letr_Y - Lebl_Y + Letl_Y - Lebr_Y) / (2.0 * (Leo_X - Lei_X))
            Rei = results.multi_face_landmarks[0].landmark[362];  # 右眼内眼角
            Rei_X, Rei_Y = int(Rei.x * w), int(Rei.y * h);
            Reo = results.multi_face_landmarks[0].landmark[263]  # 右眼外眼角
            Reo_X, Reo_Y = int(Reo.x * w), int(Reo.y * h)
            Retl = results.multi_face_landmarks[0].landmark[385]  # 右眼上左
            Retl_X, Retl_Y = int(Retl.x * w), int(Retl.y * h)
            Retr = results.multi_face_landmarks[0].landmark[387]  # 右眼上右
            Retr_X, Retr_Y = int(Retr.x * w), int(Retl.y * h)
            Rebl = results.multi_face_landmarks[0].landmark[380]  # 右眼下左
            Rebl_X, Rebl_Y = int(Rebl.x * w), int(Rebl.y * h)
            Rebr = results.multi_face_landmarks[0].landmark[373]  # 右眼下右
            Rebr_X, Rebr_Y = int(Rebr.x * w), int(Rebl.y * h)
            Rightear = (Retr_Y - Rebl_Y + Retl_Y - Rebr_Y) / (2.0 * (Rei_X - Reo_X))
            ear = (Leftear + Rightear) / 2.0
            if ear < EYE_AR_THRESH:  # 眼睛长宽比：0.2
                COUNTER += 1
                cv2.putText(im0, "Eyes closed!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
                if COUNTER >= EYE_AR_CONSEC_FRAMES:  # 阈值：3
                    TOTAL += 1
                    # speaker.Speak('闭眼')
                # 重置眼帧计数器
                COUNTER = 0
            # 进行画图操作，同时使用cv2.putText将眨眼次数进行显示
            cv2.putText(im0, "Blinks: {}".format(TOTAL), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            cv2.putText(im0, "COUNTER: {}".format(COUNTER), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            cv2.putText(im0, "EAR: {:.2f}".format(ear), (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                        2)
            Lm = results.multi_face_landmarks[0].landmark[61]  # 左侧嘴
            Lm_X, Lm_Y = int(Lm.x * w), int(Lm.y * h)
            Rm = results.multi_face_landmarks[0].landmark[291]  # 右侧嘴
            Rm_X, Rm_Y = int(Rm.x * w), int(Rm.y * h)
            Tl = results.multi_face_landmarks[0].landmark[13]  # 上嘴唇下方
            Tl_X, Tl_Y = int(Tl.x * w), int(Tl.y * h)
            Bl = results.multi_face_landmarks[0].landmark[14]  # 下嘴唇上方
            Bl_X, Bl_Y = int(Bl.x * w), int(Bl.y * h)
            mar = (Tl_Y - Bl_Y) / (Lm_X - Rm_X)
            if mar > MAR_THRESH:  # 张嘴阈值0.5
                mCOUNTER += 1
                cv2.putText(im0, "Yawning!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # 如果连续3次都小于阈值，则表示打了一次哈欠
                if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:  # 阈值：3
                    mTOTAL += 1
                    # speaker.Speak('打哈欠')
                    # 重置嘴帧计数器
                mCOUNTER = 0
            cv2.putText(im0, "Yawning: {}".format(mTOTAL), (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            cv2.putText(im0, "mCOUNTER: {}".format(mCOUNTER), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            cv2.putText(im0, "MAR: {:.2f}".format(mar), (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                        2)
        else:
            im0 = cv2.putText(im0, 'NO FACE DELECTED', (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                              (218, 112, 214), 1, 8)

        im0 = imutils.resize(im0, width=250)
        # cv2.imshow("sssss", im0)
        # cv2.waitKey(1)
        tmp = cv2.imencode("m.jpg", im0)[1]

        tmp_data = str(base64.b64encode(tmp))
        # print(tmp_data)
        # with open("ttttttttttt.txt", "w") as f:
        #     f.write(tmp_data[2:-1])
        # f.close()
        await websocket.send("data:image/jpeg;base64," + tmp_data[2:-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # cv2.waitKey(1)  # 1 millisecond

    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     print(f"Results saved to {save_dir}{s}")

    # print(f'Done. ({time.time() - t0:.3f}s)')


async def main():
    cf = configparser.ConfigParser()
    cf.read("config.ini")
    WS_HOST = cf.get("websocket", "host")
    PORT = cf.get("websocket", "port")
    WS_URL = 'ws://' + WS_HOST + ':' + str(PORT) + "/v?process"
    print(WS_URL)
    reply = "init"
    async with websockets.connect(WS_URL) as websocket:
        while len(reply) != 0:
            reply = ""
            reply = await websocket.recv()
            with torch.no_grad():
                if opt.update:  # update all models (to fix SourceChangeWarning)
                    for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                        await detect(reply, websocket)
                        strip_optimizer(opt.weights)
                else:
                    await detect(reply, websocket)


# asyncio.run(main())
# asyncio.get_event_loop().run_forever()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp4/weights/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    global opt
    opt = parser.parse_args()
    print(opt)

    # TODO: 把speaker要做的事情交给前端
    # speaker = win32com.client.Dispatch('SAPI.SPVOICE')
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # 定义两个常数
    # 眼睛长宽比
    # 闪烁阈值
    EYE_AR_THRESH = 0.2
    EYE_AR_CONSEC_FRAMES = 4
    # 打哈欠长宽比
    # 闪烁阈值
    MAR_THRESH = 0.5
    MOUTH_AR_CONSEC_FRAMES = 6
    # 初始化帧计数器和眨眼总数
    COUNTER = 0
    TOTAL = 0
    # 初始化帧计数器和打哈欠总数
    mCOUNTER = 0
    mTOTAL = 0

    mp_face_mesh = mp.solutions.face_mesh
    # help(mp_face_mesh.FaceMesh)

    model2 = mp_face_mesh.FaceMesh(
        static_image_mode=False,  # TRUE:静态图片/False:摄像头实时读取
        refine_landmarks=True,  # 使用Attention Mesh模型
        max_num_faces=9,  # 最多检测几张人脸
        min_detection_confidence=0.5,  # 置信度阈值，越接近1越准
        min_tracking_confidence=0.5,  # 追踪阈值
    )

    # 导入可视化函数和可视化样式
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    weights, view_img, save_txt, imgsz = opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave  # save inference images
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    # Set Dataloader
    vid_path, vid_writer = None, None

    check_requirements(exclude=('pycocotools', 'thop'))

    asyncio.get_event_loop().run_until_complete(main())

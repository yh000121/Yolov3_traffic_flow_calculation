import math

from sort import Sort 
import time 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from yolov3 import YOLO
from collections import deque
import pyautogui as pag
import pandas as pd
# ---------------------------------------------------#
#  初始化
# ---------------------------------------------------#
# 创建检测器
yolo = YOLO()
# 创建跟踪器
tracker = Sort()
# 生成多种不同的颜色
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype='uint8')
# 存储中心点
pts = [deque(maxlen=30) for _ in range(9999)]
# 帧率
fps = 0
# 当前运行帧数
frame_num = 1
#速度表格

# ---------------------------------------------------#
#  虚拟线圈统计车流量
# ---------------------------------------------------#
# 虚拟线圈
line = [(0, 160), (1500, 160)]
# AC = ((C[0] - A[0]), (C[1] - A[1]))
# AB = ((B[0] - A[0]), (B[1] - A[1]))
# 计算由A，B，C三点构成的向量AC，AB之间的关系
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
# 检测AB和CD两条直线是否相交
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
#计算车辆速度
def estimateSpeed(location1,location2,fps,width):
    carWidth = 1.85 # 1.85m
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = width / carWidth
    d_meters = d_pixels / ppm
    return 3.6 * d_meters * fps

# 车辆总数
counter = 0
# 正向车道的车辆数据
counter_up = 0
# 逆向车道的车辆数据
counter_down = 0

# ---------------------------------------------------#
#  读取视频并获取基本信息
# ---------------------------------------------------#
cap = cv2.VideoCapture("D:/pycharm_workspace3.7/YOLOV3-SORT-master/input/test_1.mp4")

try:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[INFO] total {} Frame in video".format(total))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("[INFO] video size :{}".format(size))
    fps_cur = int(cap.get(cv2.CAP_PROP_FPS))
    print("[INFO] video fps :{}".format(fps_cur))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter("./output/output.mp4", fourcc, fps_cur, size, True)
except:
    print("[INFO] could not determine in video")


# ---------------------------------------------------#
#  逐帧检测并追踪
# ---------------------------------------------------#
while True:
    (ret, frame) = cap.read()
    if not ret:
        break
    t1 = time.time()

    dets = yolo.predict(frame)

    if len(dets) == 0:
        continue
    else:
        tracks = tracker.update(dets)
    
    num = 0
    for track in tracks:
        bbox = track[:4] # 跟踪框坐标
        indexID = int(track[4]) # 跟踪编号
        # 保存该车的图片
        saveImg = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] # 高，宽
        # try:
            # cv2.imwrite("D:/Image/" + str(indexID) + ".jpg", saveImg)
            # cv2.imwrite("D:/Image/" + str(frame_num) + '_' + str(num) + ".jpg", saveImg)
        # except cv2.error:
            # print(cv2.error)
        # 随机分配颜色
        color = [int(c) for c in COLORS[indexID % len(COLORS)]]
        # 各参数依次是：照片/（左上角，右下角）/颜色/线宽
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
        # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
        cv2.putText(frame, str(indexID), (int(bbox[0]), int(bbox[1] - 10)), 0, 5e-1, color, 1)

        # 记录当前帧的车辆数
        num += 1
        # 检测框中心(x,y)
        center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
        pts[indexID].append(center)
        cv2.circle(frame, center, 1, color, 1)
        # 计算车速(该算法在摄像头对准车辆正面时较准，可配合OCR识别超速车牌，在摄像头广角测车流量时不适合)
        speed = estimateSpeed(center, pts[indexID][len(pts[indexID])-2], fps_cur, bbox[2]-bbox[0])
        speed = round(speed, 2) # 保留2位
        cv2.putText(frame, str(speed) + 'km/h', (int(bbox[0]+10), int(bbox[1])), 0, 3e-1, color, 1)
        # 显示运动轨迹
        for j in range(1, len(pts[indexID])):
            if pts[indexID][j - 1] is None or pts[indexID][j] is None:
                continue
            cv2.line(frame, (pts[indexID][j - 1]), (pts[indexID][j]), color, 1)

        # 虚拟线圈计数
        if len(pts[indexID]) >= 2: 
            p1 = pts[indexID][-2]
            p0 = pts[indexID][-1]
            if intersect(p0, p1, line[0], line[1]):
                counter += 1
                if p1[1] > p0[1]:
                    counter_down += 1
                else:
                    counter_up += 1
    # 显示结果
    cv2.line(frame, line[0], line[1], (0, 255, 0), 1)
    cv2.putText(frame, str(counter), (20, 90), 0, 0.8, (255, 0, 0), 1)# 车辆总数
    cv2.putText(frame, str(counter_up), (200, 90), 0, 0.8, (0, 255, 0), 1)# 正向车道的车辆数据
    cv2.putText(frame, str(counter_down), (450, 90), 0, 0.8, (0, 0, 255), 1)# 逆向车道的车辆数据
    cv2.putText(frame, "Current Car Counter: " + str(num), (int(20), int(40)), 0, 5e-1, (0, 255, 0), 1)
    cv2.putText(frame, "FPS: %f" %(fps), (int(20), int(20)), 0, 5e-1, (0,255,0), 1)
    cv2.namedWindow("YOLOV3-SORT", 0)
    cv2.resizeWindow('YOLOV3-SORT', 1280, 720)
    # 计算帧率
    fps = (fps + (1. / (time.time() - t1))) / 2
    writer.write(frame)
    cv2.imshow('YOLOV3-SORT', frame)
    # 保存每一帧
    cv2.imwrite("D:/Image/"+str(frame_num)+".jpg", frame)
    frame_num = frame_num + 1
    # Q键停止
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        print('stop')
        break
# 释放资源
writer.release()
cap.release()
cv2.destroyAllWindows()
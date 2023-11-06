import cv2
import mediapipe as mp
import time

# （1）视频捕获
cap = cv2.VideoCapture(4)  # 0代表电脑自带的摄像头

# （2）创建检测手部关键点的方法
mpHands = mp.solutions.hands  # 接收方法
hands = mpHands.Hands(
    static_image_mode=False,  # 静态追踪，低于0.5置信度会再一次跟踪
    max_num_hands=2,  # 最多有2只手
    min_detection_confidence=0.5,  # 最小检测置信度
    min_tracking_confidence=0.5,
)  # 最小跟踪置信度

# 创建检测手部关键点和关键点之间连线的方法
mpDraw = mp.solutions.drawing_utils

# 查看时间
pTime = 0  # 处理一张图像前的时间
cTime = 0  # 一张图处理完的时间

# （3）处理视频图像
while True:  # 对每一帧视频图像处理
    # 返回是否读取成功和读取的图像
    success, img = cap.read()

    # 在循环中发送rgb图像到hands中，opencv中图像默认是BGR格式
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 把图像传入检测模型，提取信息
    results = hands.process(imgRGB)

    # 检查是否检测到什么东西了，没有检测到手就返回None
    # print(results.multi_hand_landmarks)

    # 检查每帧图像是否有多只手，一一提取它们
    if results.multi_hand_landmarks:  # 如果没有手就是None
        for handlms in results.multi_hand_landmarks:
            # 绘制每只手的关键点
            mpDraw.draw_landmarks(
                img, handlms, mpHands.HAND_CONNECTIONS
            )  # 传入想要绘图画板img，单只手的信息handlms
            # mpHands.HAND_CONNECTIONS绘制手部关键点之间的连线

    # 记录执行时间
    cTime = time.time()
    # 计算fps
    fps = 1 / (cTime - pTime)
    # 重置起始时间
    pTime = cTime

    # 把fps显示在窗口上；img画板；取整的fps值；显示位置的坐标；设置字体；字体比例；颜色；厚度
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 显示图像
    cv2.imshow("Image", img)  # 窗口名，图像变量
    if cv2.waitKey(1) & 0xFF == 27:  # 每帧滞留1毫秒后消失；ESC键退出
        break

# 释放视频资源
cap.release()
cv2.destroyAllWindows()

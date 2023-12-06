import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import math
import time
from matplotlib import pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

same_img_path = "./test"
if not os.path.exists(same_img_path):
    os.mkdir(same_img_path)
def get_timestamp():
    return time.strftime("%Y%m%d%H%M%S")
timestamp = get_timestamp()
seq = 0
    
model_path = "./exported_model/gesture_recognizer.task"

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# 创建检测手部关键点和关键点之间连线的方法
mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands  # 接收方法
def print_result(
    result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int
):    
    
    if len(result.gestures) > 0:
        print(f"gestures len:{len(result.gestures)}, handedness len:{len(result.handedness)}, result.hand_landmarks len:{len(result.hand_landmarks)}")
    for gesture in result.gestures:
        print(gesture)
    for handedness in result.handedness:
        print(handedness)  
    
  
    for hand_landmarks in result.hand_landmarks:
        print(f"Lhand_landmarks len:{len(hand_landmarks)}")
        # for hand_landmark in hand_landmarks:
        #     print(hand_landmark)
    #     print(hand_landmarks[0].x)
    #     print(hand_landmarks[8].x)
    #     if hand_landmarks[0].category_name == "right" or hand_landmarks[0].ca tegory_name == "left":
    #         if hand_landmarks[0].x > hand_landmarks[8].x:
    #             print("Right")
    #         else:
    #             print("Left")
    #             # cv2.putText(output_image, "Right", (10, 100), cv2.FONT_HERSH_SIMPLEX, 2, (255, 0, 0), 2)
    #             # cv2.imshow("output", output_image)
    #             # cv2.waitKey(1)
           
        
        
        

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    # running_mode=VisionRunningMode.LIVE_STREAM,
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    # result_callback=print_result,
)
with GestureRecognizer.create_from_options(options) as recognizer:
    # The detector is initialized. Use it here.

    # Open webcam
    # Setup computer vision for live webcam
    cap = cv2.VideoCapture(0)  # uncomment for live webcam video analysis
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # Set width of camera
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Set heigh of camera
    font = cv2.FONT_HERSHEY_SIMPLEX  # Set font for computer vision
    # 查看时间
    pTime = 0  # 处理一张图像前的时间
    cTime = 0  # 一张图处理完的时间
    while cap.isOpened():  # as long as the webcam is open this will happen
        ret, frame = cap.read()  # Saves the image from your webcam as a frame
        # 水平镜像
        frame = cv2.flip(frame, 1)
        # OPENCV reads in as BGR.  This line recolors image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image.flags.writeable = False  # Saves memory by making image not writeable
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        # Recolor image back to BGR for OPENCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        numpy_frame_from_opencv = np.array(image)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv
        )

        #recognizer.recognize_asnmp_image, frame_timestamp_ms)
        gesture_recognition_result = recognizer.recognize_for_video(mp_image, frame_timestamp_ms)
        # print(len(gesture_recognition_result.hand_landmarks))        
        if len(gesture_recognition_result.hand_landmarks) > 0: 
            # print(gesture_recognition_result.gestures)    
            for i,hand_landmarks in enumerate(gesture_recognition_result.hand_landmarks):
                # print(len(hand_landmarks))
                # print(hand_landmarks)
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])

                mp_drawing.draw_landmarks(
                image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())      
                      
                # print(i)          
                # print(gesture_recognition_result.gestures[i][0].category_name)
                if gesture_recognition_result.gestures[i][0].category_name == "right" or gesture_recognition_result.gestures[i][0].category_name == "left":                                
                    if hand_landmarks[0].x > hand_landmarks[8].x:
                        gesture_recognition_result.gestures[i][0].category_name ="left"
                    else:
                        gesture_recognition_result.gestures[i][0].category_name = "right"
                if  gesture_recognition_result.gestures[i][0].category_name is not None:       
                # if  gesture_recognition_result.gestures[i][0].category_name is not None and gesture_recognition_result.gestures[i][0].score >= 0.800: 
                    # print(gesture_recognition_result)
                    cv2.putText(image, f'{gesture_recognition_result.gestures[i][0].category_name} {round(gesture_recognition_result.gestures[i][0].score,3)}', (int(hand_landmarks[0].x* width)-30 , int(hand_landmarks[0].y*height)+20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
              
        # 记录执行时间
        cTime = time.time()
        # 计算fps
        fps = 1 / (cTime - pTime)
        # 重置起始时间

        pTime = cTime

        # 把fps显示在窗口上；img画板；取整的fps值；显示位置的坐标；设置字体；字体比例；颜色；厚度
        # cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)     
        cv2.imshow("Vedio", image)

        key = cv2.waitKey(10)
        if key & 0xFF == ord(
            "q"
        ):  # This puts you out of the loop above if you hit q
            break
        
        if key & 0xFF == ord("s"):
            seq = seq + 1
            cv2.imwrite(f"{same_img_path}/{timestamp}_{seq}.jpg", frame)
            print("save", f"{same_img_path}/{timestamp}_{seq}.jpg")

    cap.release()  # Releases the webcam from your memory
    cv2.destroyAllWindows()  # Closes the window for the webcam

import cv2
import time
import os

import shutil


path = "./data/gestrue/left"


def get_timestamp():
    return time.strftime("%Y%m%d%H%M%S")

#shutil.rmtree(path)
#os.mkdir(path)

if not os.path.exists(path):
    os.mkdir(path)

seq = 0
timestamp = get_timestamp()
cap = cv2.VideoCapture(4)
while True:
    success, img = cap.read()
    if not success:
        print("read failed")
        continue

    cv2.imshow("Image", img)

    key = cv2.waitKey(100)
    if key & 0xFF == ord("s"):
        seq = seq + 1
        cv2.imwrite(f"{path}/{timestamp}_{seq}.jpg", img)
        print("save", f"{path}/{timestamp}_{seq}.jpg")
    if key & 0xFF == 27:
        break

# 释放视频资源
cap.release()
cv2.destroyAllWindows()

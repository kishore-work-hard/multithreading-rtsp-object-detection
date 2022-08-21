import cv2
import os, time
import threading
import matplotlib.pyplot as plt


RTSP_URL = 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4'
streams=(
    [RTSP_URL,'cam1'],
    [RTSP_URL,'cam2'],
    [RTSP_URL,'cam3'],
)
coco_classes = ["car", "plate", "motorcycle"]
net = cv2.dnn.readNet("../../RequiredFiles/weights/custom-yolo.weights","../../RequiredFiles/cfg/custom-yolo.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)
# FrameData = []


def object_detection(frame):
    c, v, b = model.detect(frame, 0.2, 0.4)
    c = [coco_classes[xc] for xc in c]
    for (classid, score, box) in zip(c, v, b):
        if classid == 0 or classid == 2:
            lx, ly, cw, ch = box
        xc = cv2.rectangle(frame, box, (255, 0, 255), 3)
        # plt.imshow(cv2.cvtColor(xc, cv2.COLOR_BGR2RGB))
        # plt.waitforbuttonpress()


def cams(s):
    url = s[0]
    cam = s[1]

    video = cv2.VideoCapture(url)
    while True:
        FrameData = []
        _, frame = video.read()
        t = time.time()

        # adding frame, cam name and time to list
        FrameData.append(frame)
        FrameData.append(cam)
        FrameData.append(t)

        object_detection(FrameData[0])

        cv2.imshow(cam, frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()




thread_list = []
for s in streams:
    x = threading.Thread(target=cams, args=(s,))
    thread_list.append(x)


for thread in thread_list:
    thread.start()

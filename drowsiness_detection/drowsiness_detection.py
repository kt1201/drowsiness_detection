# 졸음운전 방지장치
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])

    C = euclidean_dist(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

# --cascade : 얼굴 탐지에 사용되는 Haarcascade XML 파일의 경로
# --shape-predictor : dlib facial landmark predictor 파일의 경로
# --alarm : 졸음이 감지될 때 pygame 소리의 사용 여부
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True)
ap.add_argument("-p", "--shape-predictor", required=True)
ap.add_argument("-a", "--alarm", type=int, default=0)
args = vars(ap.parse_args())

if args["alarm"] > 0:
    import pygame
    pygame.init()
    soundObj = pygame.mixer.Sound('wakeup.wav')

os.system('sudo /home/pi/PiBits/ServoBlaster/user/servod --pcm')

EYE_AR_THRESH = 0.28
EYE_AR_CONSEC_FRAMES = 10

COUNTER = 0
ALARM_ON = False

detector = cv2.CascadeClassifier(args["cascade"])
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# 비디오 스트림 스레드 시작
print("[INFO] starting video stream thread...")
vs = VideoStream(src=-1).start()
#vs = VideoStream(usePicamera=True).start()
time.sleep(1.0) # 카메라 워밍업

# 비디오 스트리밍의 프레임 loop
while True:
    # 스레드된 비디오 파일 스트림에서 프레임을 가져와
    # 크기를 조정하고, 흑백으로 변환
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 흑백 영상 프레임에서 얼굴 감지
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
    minNeighbors=5, minSize=(30, 30), 
    flags=cv2.CASCADE_SCALE_IMAGE)

    # 얼굴 감지 loop
    for (x, y, w, h) in rects:
        # Haarcascade bounding box에서 dlib 사각형 객체 구성
        rect = dlib.rectangle(int(x), int(y), int(x + w),
            int(y + h))

        # facial landmark 결정 후
        # facial landmark(x, y)좌표를 NumPy 배열로 변환
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 좌우 눈 좌표를 추출한 후
        # 좌표를 이용하여 양쪽 눈의 가로세로 비율을 계산한다.
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # 양쪽 눈의 가로세로 비율 평균
        ear = (leftEAR + rightEAR) / 2.0

        # 각 눈의 convexHull(볼록다각형)을 계산한 다음,
        # 각각의 눈을 시각화 한다.
        # 스크린 없는 임베디드 제품을 만든다면 필요하지 않다.
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 눈 가로세로 비율이 깜박임 임계값 미만인지 확인하고,
        # 눈 깜박임 프레임 카운터 증가
        if ear < EYE_AR_THRESH:
            COUNTER += 1


            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    if args["alarm"] > 0:
                        soundObj.play()
                        os.system("echo 1=150 > /dev/servoblaster")
                        os.system("echo 2=50 > /dev/servoblaster")
                        time.sleep(1)
                        os.system("echo 1=50 > /dev/servoblaster")
                        os.system("echo 2=150 > /dev/servoblaster")
                        time.sleep(9)
                        soundObj.stop()

                # 프레임에 알람 표시
                cv2.putText(frame, "drowsiness detect!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False

        # 프레임에 계산된 눈 가로세로 비율을 표시
        cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 프레임 출력
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # 'q'를 누르면, loop에서 벗어남
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

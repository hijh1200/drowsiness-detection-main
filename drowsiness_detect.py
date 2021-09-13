'''This script detects if a person is drowsy or not,using dlib and eye aspect ratio
calculations. Uses webcam video feed as input.'''

# Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame  # For playing sound
import time
import dlib
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize Pygame and load music(wav 파일 지원)
pygame.mixer.init()  # 믹서 모듈의 초기화 함수
pygame.mixer.music.load('audio/alert.wav')  # 음악 로딩
guideSound = pygame.mixer.Sound("audio/start_guide_wav.wav")  # 안내 음성
tuningSound = pygame.mixer.Sound("audio/tuning_1.wav")  # 튜닝 음성

# Minimum threshold of eye aspect ratio below which alarm is triggerd
# 눈의 EAR의 THRESH 기본값을 0.3으로 설정
eye_sum = 0
# 졸음운전을 판단할 때 사용하는 임곗값(눈)
EYE_ASPECT_RATIO_THRESHOLD = 0
# 하품을 판단할 때 사용하는 임곗값(입)
MOUTH_THRESHOLD = 0
# 고개숙임 판단 시 사용하는 임계값
HEAD_DOWN_THRESHOLD = 0
# 눈 깜빡임 기준 횟수
EYE_STANDARD_NUMBER = 0

# Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 5  # 연속 눈 감기 검출을 위한 기준 시간(프레임)
MOUTH_FRAMES = 10  # 하품 검출위한 기준 시간(프레임)
TUNING_FRAMES = 60  # 사용자별 튜닝 기준 시간
EYE_STANDARD_TIME = 40  # 정상적인 눈깜빡임 빈도 계산을 위한 기준 시간

# 눈깜빡임 여부 리스트(0: 눈뜸, 눈감음)
eye_numList = [0 for i in range(EYE_STANDARD_TIME)]  # 전부 0으로 초기화
eye_diff = [0 for i in range(2)]  # 전부 0으로 초기화
# 눈깜빡임 패턴
eye_pattern = [0 for i in range(3)]
# 이전 EAR값
prev_eyeRate = 0
# 정상적인 눈 깜빡임 횟수
eye_number = 0
# 눈 감았을 때 ear 값
eye_close = 0

# COunts no. of consecutuve frames below threshold value
# 프레임 카운터(눈) 초기화
EYE_COUNTER = 0

# 프레임 카운터(입) 초기화
COUNTER_MOUTH = 0

# Load face cascade which will be used to draw a rectangle around detected faces.
# 얼굴 인식: 정면 얼굴에 대한 Haar_Cascade 학습 데이터 (직사각형 그리는 용도)
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")


# 눈 종횡비(EAR) 계산 함수
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])  # 두 점사이의 거리를 구하는데 사용하는 함수
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


# 입 거리비 계산 함수
def mouth_rate(lip):
    leftPoint = lip[0]  # 48
    rightPoint = lip[6]  # 54
    topPoint = lip[3]  # 51
    rowPoint = lip[9]  # 57

    mouthWidth = distance.euclidean(leftPoint, rightPoint)  # 입 너비
    mouthHeight = distance.euclidean(topPoint, rowPoint)  # 입 높이

    mouthRate = mouthHeight / mouthWidth

    return mouthRate


# 고개 숙임 계산 함수
def head_rate(head):
    headX = distance.euclidean(head[27], head[30])  # 코 시작~끝(27, 30)
    headY = distance.euclidean(head[30], head[8])  # 코 끝~턱 끝(30, 8)

    headRate = headX / headY

    return headRate


# Load face detector and predictor, uses dlib shape predictor file
# 68개의 얼굴 랜드마크 추출
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Extract indexes of facial landmarks for the left and right eye
# 위에서 추출한 랜드마크에서 왼쪽 눈과 오른쪽 눈의 랜드마크 좌표 추출
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
# 랜드마크에서 입 좌표 추출
(mStart, mEnd) = 48, 68

video_file = "./video/eye_test.mp4"
# Start webcam video capture
# 첫번째(0) 카메라를 VideoCapture 타입의 객체로 얻어옴\
video_capture = cv2.VideoCapture(0)
# video_capture = cv2.VideoCapture(video_file)


# Give some time for camera to initialize(not required)
time.sleep(2)

# 상태(0: 기기 부팅 초기상태, 1: s를 눌러 평상시 값 3개 측정 단계, 2: 측정 후 평균 구하는 단계, 3: 졸음 판별 단계)
state = 0
# 졸음 단계(0:평상시 상태, 1: 졸음 전조 단계, 2: 졸음단계)
drowsiness_level = 0

# 현재시각
currentTime = time.strftime("%Y%m%d_%H%M%S")
outputFileName = "./output/" + currentTime + ".avi"
outputImageName = "./graphImage/" + currentTime + ".png"

# 웹캠의 속성 값을 받아오기
# 정수 형태로 변환하기 위해 round
w = round(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)  # 카메라에 따라 값이 정상적, 비정상적

print(fps)  # 30.0

# fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# 1프레임과 다음 프레임 사이의 간격 설정
delay = round(1000 / fps)
print(delay)

# 웹캠으로 찰영한 영상을 저장하기
# cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
out = cv2.VideoWriter(outputFileName, fourcc, fps / 10, (w, h))

X = 0

# 실시간 그래프 그리기 위함
# plt.figure(1, [0,0])

# 그래프 그리기위한 값들을 담은 리스트
# dataX = []
dataY = [[], [], []]  #

# 평균 값 측정을 위한 값들을 담아놓는 리스트
eyeList = []
mouthList = []
headList = []

startTime = datetime.now()

FRAME_COUNTER = 0  # 프레임 카운터
THIRD_FRAME = 0  # 3단계 첫 시작 판별을 위한 카운터
ax = []  # 그래프 여러개 그리기 위한 Axes
graphTitle = ["eye", "mouth", "head"]

FIRST_YAWN = 0  # 첫 번째 하품
SECOND_YAWN = 0 # 두 번째 하품

while (True):
    # Read each frame and flip it, and convert to grayscale
    # ret : frame capture결과(boolean)
    # frame : Capture한 frame
    ret, frame = video_capture.read()  # 비디오 읽기
    frame = cv2.flip(frame, 1)  # 프레임 좌우반전(flipcode > 0)
    # BGR 이미지: / GrayScale 이미지: 색상정보X,밝기 정보로만 구성(0~255, 검~흰)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # color space 변환(BGR->Grayscale로 변환)

    # Detect facial points through detector function
    faces = detector(gray, 0)

    # Detect faces through haarcascade_frontalface_default.xml
    # grayscale 이미지를 입력해 얼굴 검출하여 위치를 리스트로 리턴(x, y, w, h)
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)  # (ScaleFactor, minNeighbor)

    # 블랙박스처럼 현재 시각 나타내기
    currentTime = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, currentTime, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 키보드 입력으로 시작하기
    if (state == 0):
        if (cv2.waitKey(1) == ord('s')):
            state = 1  # 시작 상태로 변경
            guideSound.play()  # 안내음성 출력
            time.sleep(5)
            startTime = datetime.now()  # 현재 시각 저장

    if (state != 0):
        """
        # 얼굴 사각형 그리기
        # (x, y): 좌상단 위치, (w, h): 이미지 크기
        for (x, y, w, h) in face_rectangle:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        """
        # Detect facial points
        # 검출된 얼굴 들에 대한 반복
        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # 얼굴 랜드마크 68 point 찍기
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            # 왼쪽,오른쪽 눈 index 리스트로 얻기
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            # 입 index 리스트로 얻기
            innerMouth = shape[mStart:mEnd]

            # 양쪽 눈의 EAR 계산
            leftEyeAspectRatio = eye_aspect_ratio(leftEye)
            rightEyeAspectRatio = eye_aspect_ratio(rightEye)

            # 입 거리비 계산
            mouthRate = round(mouth_rate(innerMouth), 3)

            # 고개 숙임 Rate 계산
            headRate = round(head_rate(shape), 3)

            # 양쪽 눈 EAR 값의 평균 계산
            eyeAspectRatio = round((leftEyeAspectRatio + rightEyeAspectRatio) / 2, 3)

            # Use hull to remove convex contour discrepencies and draw eye shape around eyes
            # 눈 윤곽선에서 블록껍질 검출(경계면 둘러싸는 다각형 구하기)
            # convexHull(윤곽선:윤곽선 검출 함수에서 반환되는 구조, 방향:True/False[시계/반시계])
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            # 입 윤곽선에서 블록껍질 검출
            innerMouthHull = cv2.convexHull(innerMouth)

            # 눈 경계선 그리기
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            # 입 경계선 그리기
            cv2.drawContours(frame, [innerMouthHull], -1, (0, 255, 0), 1)

            # 좌측 상단 기준 text 표시
            # 상태 표시
            cv2.putText(frame, "drowsy_level : {:d}".format(drowsiness_level), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 30, 20), 2)

            # 눈 EAR 표시
            cv2.putText(frame, "EAR : {:.3f}/{:.3f}".format(eyeAspectRatio, EYE_ASPECT_RATIO_THRESHOLD), (0, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 30, 20), 2)

            # 입 거리비 표시
            cv2.putText(frame, "mouthRate : {:.3f}/{:.3f}".format(mouthRate, MOUTH_THRESHOLD), (0, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 30, 20), 2)

            # 고개 숙임비 표시
            cv2.putText(frame, "headRate : {:.3f}/{:.3f}".format(headRate, HEAD_DOWN_THRESHOLD), (0, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 30, 20), 2)

            # 눈깜빡임 횟수 표시
            cv2.putText(frame, "blink : {:d}/{:.1f}".format(eye_number, EYE_STANDARD_NUMBER), (0, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 30, 20), 2)



            # # 현재 프레임 출력
            # cv2.putText(frame, "THIRD_FRAME : {:d}".format(THIRD_FRAME), (0, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (200, 30, 20), 2)
            # # 첫 번째 하품을 한 시점의 프레임 출력
            # cv2.putText(frame, "FIRST_YAWN : {:d}".format(FIRST_YAWN), (0, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (200, 30, 20), 2)
            # # 두 번째 하품을 한 시점의 프레임 출력
            # cv2.putText(frame, "SECOND_YAWN : {:d}".format(SECOND_YAWN), (0, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (200, 30, 20), 2)



            """
            nowTime = datetime.now()        # 현재 시각
            diffTime = nowTime - startTime  # 측정 시간
            """
            # 20프레임 값 측정 후
            if (state == 1 and FRAME_COUNTER >= TUNING_FRAMES):
                state = 2  # 측정값들의 평균(임계값) 구하는 단계

            if (state == 3):
                THIRD_FRAME += 1
                frameIdx = (THIRD_FRAME - 1) % EYE_STANDARD_TIME
                diffIdx = (THIRD_FRAME - 1) % 3
                # 실시간 그래프그리기
                for i in range(0, 3):
                    if (THIRD_FRAME == 1):
                        ax.append(plt.subplot(2, 2, i + 1))  # 그래프 추가(행, 열, 위치)
                        plt.title(graphTitle[i])  # 그래프 제목
                        plt.xticks(rotation=45)  # x축 라벨 -45
                    X += 1
                    if (i == 0):
                        data = eyeAspectRatio  # 눈
                        # prev_eyeRate = eyeAspectRatio
                        eye_pattern[diffIdx] = eyeAspectRatio
                    if (i == 1):
                        data = mouthRate  # 입
                    if (i == 2):
                        data = headRate  # 고개숙임
                    dataY[i].append(data)  # 데이터 넣기
                    # plt.scatter(X, eyeAspectRatio)
                    ax[i].plot(dataY[i])

                plt.tight_layout()  # 그래프 겹치지 않게 Axes 조절
                plt.pause(0.000000001)

                # Detect if eye aspect ratio is less than threshold
                # 졸음판단(현재 EAR이 임계값보다 작은지 확인)
                # 정상적인 눈깜빡임 빈도(횟수) 계산(이전값2, 이전값1, 현재값)
                if (THIRD_FRAME > 3):
                    eye_diff[0] = eye_pattern[diffIdx - 1] - eye_pattern[diffIdx - 2]
                    eye_diff[1] = eye_pattern[diffIdx] - eye_pattern[diffIdx - 1]
                # print(eye_pattern)

                # 정상적인 눈깜빡임 검출
                if (eye_pattern[diffIdx - 1] < 0.9 * EYE_ASPECT_RATIO_THRESHOLD):
                    EYE_COUNTER += 1
                    # 눈감았을 때 ear 값
                    eye_close = eye_pattern[diffIdx - 1]
                    eye_close_ratio = round(eye_close / EYE_ASPECT_RATIO_THRESHOLD, 2)  # 눈을 감은 정도

                    # cv2.putText(frame, "blink", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                    # eye_numList[frameIdx] = 1 # 눈깜빡임이 일어나면 1로
                    eye_numList[frameIdx] = eye_close_ratio  # 눈 감은 정도(0~1: 0에 가까울 수록 많이 감은 것)
                else:
                    EYE_COUNTER = 0
                    eye_numList[frameIdx] = 0  # 눈 뜸

                eye_number = EYE_STANDARD_TIME - eye_numList.count(0)  # 눈깜빡임 횟수 카운팅

                # print(eye_numList)  # 기준 시간 내 눈을 몇번 얼마나 감았는지 기록

                # 졸음 단계 변경 부분(2프레임에 한번씩 검사)
                if (THIRD_FRAME % 2 == 0):
                    # 연속 눈 감기(10Frame)
                    if (EYE_COUNTER > EYE_ASPECT_RATIO_CONSEC_FRAMES):
                        print("눈감음")
                        drowsiness_level = 2  # 졸음 단계로 변경
                    elif (drowsiness_level == 2 and EYE_COUNTER == 0):
                        # 눈 감고 있다가 뜰 경우 바로 평상시 단계로 변경하기 위해
                        drowsiness_level = 0  # 평상시 단계로 변경
                    elif (eye_number >= EYE_STANDARD_NUMBER * 1.3):
                        drowsiness_level = 1  # 졸음 전조 단계로 변경




###################################



                if (mouthRate > MOUTH_THRESHOLD * 2):
                    COUNTER_MOUTH += 1

                    # 하품 조건을 만족하고 이미 첫 번째 하품을 한 상태라면 SECOND_YAWN 에 현재 프레임 입력
                    if (COUNTER_MOUTH >= MOUTH_FRAMES and FIRST_YAWN != 0 and THIRD_FRAME - FIRST_YAWN > 200):
                        SECOND_YAWN = THIRD_FRAME

                    # 하품 조건을 만족하고 첫 번째 하품을 한 적이 없다면 FIRST_YAWN 에 현재 프레임 입력
                    elif(COUNTER_MOUTH >= MOUTH_FRAMES and FIRST_YAWN == 0):
                        FIRST_YAWN = THIRD_FRAME
                else:
                    COUNTER_MOUTH = 0


                # 첫 번째 하품을 하고 나서 3분 이내에 두 번째 하품을 했다면 LEVEL 1
                if(SECOND_YAWN - FIRST_YAWN > 200 and SECOND_YAWN - FIRST_YAWN < 3600):
                    drowsiness_level = 1


                # 첫 번째 하품을 한지 3분(3600프레임)이 지나고도 두 번째 하품을 하지 않는다면 FIRST_YAWN 은 0으로 초기화
                if(THIRD_FRAME - FIRST_YAWN > 3600):
                     FIRST_YAWN = 0




###################################






                # 졸음 단계별 사운드 조절
                if (drowsiness_level == 0):
                    pygame.mixer.music.stop()  # 소리 출력 정지
                elif (drowsiness_level == 1):
                    # 졸음 전조 단계이면
                    # pygame.mixer.music.play(-1)
                    cv2.putText(frame, "1 LEVEL", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                elif (drowsiness_level == 2):
                    # 졸음 단계 이면
                    pygame.mixer.music.play(-1)
                    cv2.putText(frame, "2 LEVEL", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

                """
                if (eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD * 0.9):
                    # 임계값보다 EAR 작으면(눈 감고 있는 상태)
                    EYE_COUNTER += 1
                    # If no. of frames is greater than threshold frames,
                    if EYE_COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                        # 눈을 깜빡인건지 졸고있는건지 확인
                        pygame.mixer.music.play(-1)
                        cv2.putText(frame, "You are close eyes", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                else:
                    # 임계값보다 EAR 크면(눈 뜨고 있는 상태)
                    pygame.mixer.music.stop()  # 소리 출력 정지
                    EYE_COUNTER = 0  # 카운터 초기화

                # 하품판단
                if (mouthRate > MOUTH_THRESHOLD * 2):
                    COUNTER_MOUTH += 1
                    # 10 프레임 연속 조건 충족 시, 하품 판단
                    if (COUNTER_MOUTH >= MOUTH_FRAMES):
                        cv2.putText(frame, "You are Yawn", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                else:
                    COUNTER_MOUTH = 0

                # 고개 숙임 판단
                if (headRate > HEAD_DOWN_THRESHOLD * 1.8):
                    cv2.putText(frame, "You are HEAD DOWN", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                """

            elif (state == 1):
                cv2.rectangle(frame, (200, 10), (250, 40), (0, 0, 255), 2)
                cv2.putText(frame, "ing!".format(headRate), (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
                tuningSound.play()  # 측정 음성

                # 임계값 측정 단계이면
                FRAME_COUNTER += 1  # 프레임 수 세기
                eyeList.append(round(eyeAspectRatio, 3))
                mouthList.append(round(mouthRate, 3))
                headList.append(round(headRate, 3))

                if (FRAME_COUNTER >= 2):
                    now_idx = FRAME_COUNTER - 1

                    # 증감율 계산
                    eye_diff[0] = (eyeList[now_idx] - eyeList[now_idx - 1]) / eyeList[now_idx - 1]

                    # 눈깜빡임 횟수 및 EAR 합계 구하기
                    if (eye_diff[0] < -0.2):
                        EYE_STANDARD_NUMBER += 1  # 눈 깜빡임 횟수 증가
                    else:
                        eye_sum += eyeList[now_idx]  # ear 합계 구하기

            elif (state == 2):
                """
                # 눈깜빡인 값 제외하고 계산
                for i in range(TUNING_FRAMES - 3):
                    # 증감율 계산
                    eye_diff[0] = (eyeList[i + 1] - eyeList[i])/ eyeList[i]
                    eye_diff[1] = (eyeList[i + 2] - eyeList[i + 1])/ eyeList[i + 1]

                    if (eye_diff[0] < -0.2):
                        EYE_STANDARD_NUMBER += 1 # 눈 깜빡임 횟수 증가
                    else:
                        EYE_ASPECT_RATIO_THRESHOLD += eyeList[i + 1]
                """
                # 평균값(임계값 구하기)
                EYE_ASPECT_RATIO_THRESHOLD = round(eye_sum / (len(eyeList) - EYE_STANDARD_NUMBER), 3)
                # EYE_ASPECT_RATIO_THRESHOLD = round(sum(eyeList) / len(eyeList), 3)
                # 프레임수에 대한 눈깜빡임 수 구하기(튜닝시간:튜닝동안 눈깜빡임수 = 눈깜빡임 기준 시간:x)
                EYE_STANDARD_NUMBER = round(EYE_STANDARD_NUMBER * EYE_STANDARD_TIME / TUNING_FRAMES, 1)

                MOUTH_THRESHOLD = round(sum(mouthList) / len(mouthList), 3)
                HEAD_DOWN_THRESHOLD = round(sum(headList) / len(headList), 3)
                state = 3  # 졸음 판별 단계로 바꿈

                print(EYE_ASPECT_RATIO_THRESHOLD, MOUTH_THRESHOLD, HEAD_DOWN_THRESHOLD)

    # 비디오 저장
    out.write(frame)  # 영상 데이터만 저장. 소리는 X

    # 비디오 보여주기
    cv2.moveWindow(winname='Video', x=0, y=100)  # 특정 위치에 띄우기
    cv2.imshow('Video', frame)

    # 키보드 입력으로 중지시키기
    if (cv2.waitKey(1) == ord('q')):
        state = 4  # 종료단계
        plt.savefig(outputImageName)  # 그래프 이미지 저장
        # plt.show()
        break

# Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()
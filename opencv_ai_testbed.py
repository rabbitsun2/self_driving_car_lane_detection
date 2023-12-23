# Project: OpenCV, tensorflow 2.4 이상 Lane Detection 프로젝트(3단계:적용)
# Filename: car_ai.py
# Created Date: 2023-12-08(금)
# Author: 대학원생 석사과정 정도윤
# Description:
# 1. tensorflow 2.4버전 이상으로 형식 변경
# 2. 
#
# Reference:
# 1. https://github.com/aruneshmee/Self-Driving-Car, Accessed by 2023-12-23.
# 2. AI 인공지능 자율주행 자동차 만들기 + 데이터 수집·학습 + 딥러닝 with 라즈베리파이, 장문철, 앤써북, 2021-08-30.
#
import threading
import time
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model


def DetectLineSlope(src):
    # 흑백화
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 모서리 검출
    canny_edges = cv2.Canny(gray, 50, 200, None, 3)

    # 관심 구역 설정
    height = canny_edges.shape[0]
    rectangle = np.array([[(0, height), (120, 300), (520, 300), (640, height)]])
    mask = np.zeros_like(canny_edges)
    cv2.fillPoly(mask, rectangle, 255)
    masked_image = cv2.bitwise_and(canny_edges, mask)
    color_canny_edges = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)

    # 직선 검출
    detected_lines = cv2.HoughLinesP(masked_image, 1, np.pi / 180, 20, minLineLength=10, maxLineGap=10)

    # line color
    # color = [0, 0, 255]
    # thickness = 5
    # for line in line_arr:
    #   for x1, y1, x2, y2 in line:
    #        cv2.line(ccan, (x1, y1), (x2, y2), color, thickness)

    # 중앙을 기준으로 오른쪽, 왼쪽 직선 분리
    right_lines = np.empty((0, 5), int)
    left_lines = np.empty((0, 5), int)

    if detected_lines is not None:
        line_arr2 = np.empty((len(detected_lines), 5), int)

        for i in range(0, len(detected_lines)):
            temp = 0
            l = detected_lines[i][0]
            line_arr2[i] = np.append(detected_lines[i], np.array((np.arctan2(l[1] - l[3], l[0] - l[2]) * 180) / np.pi))

            if line_arr2[i][1] > line_arr2[i][3]:
                temp = line_arr2[i][0], line_arr2[i][1]
                line_arr2[i][0], line_arr2[i][1] = line_arr2[i][2], line_arr2[i][3]
                line_arr2[i][2], line_arr2[i][3] = temp
            if line_arr2[i][0] < 320 and (abs(line_arr2[i][4]) < 170 and abs(line_arr2[i][4]) > 95):
                left_lines = np.append(left_lines, line_arr2[i])
            elif line_arr2[i][0] > 320 and (abs(line_arr2[i][4]) < 170 and abs(line_arr2[i][4]) > 95):
                right_lines = np.append(right_lines, line_arr2[i])

    left_lines = left_lines.reshape(int(len(left_lines) / 5), 5)
    right_lines = right_lines.reshape(int(len(right_lines) / 5), 5)

    # 중앙과 가까운 오른쪽, 왼쪽 선을 최종 차선으로 인식
    try:
        left_lines = left_lines[left_lines[:, 0].argsort()[-1]]
        degree_L = left_lines[4]
        cv2.line(color_canny_edges, (left_lines[0], left_lines[1]), (left_lines[2], left_lines[3]), (255, 0, 0), 10, cv2.LINE_AA)
    except:
        degree_L = 0
    try:
        right_lines = right_lines[right_lines[:, 0].argsort()[0]]
        degree_R = right_lines[4]
        cv2.line(color_canny_edges, (right_lines[0], right_lines[1]), (right_lines[2], right_lines[3]), (255, 0, 0), 10, cv2.LINE_AA)
    except:
        degree_R = 0

    # 원본에 합성
    mimg = cv2.addWeighted(src, 1, color_canny_edges, 1, 0)
    return mimg, degree_L, degree_R

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.resize(image, (400, 120))
    image = cv2.GaussianBlur(image,(5, 5),0)
    _,image = cv2.threshold(image,160, 255, cv2.THRESH_BINARY_INV)
    image = image / 255
    return image

def main():

    i = 0
    cap = cv2.VideoCapture("challenge.mp4")
    carState = "stop"
    #filepath = "/home/pi/AI_CAR/video/train"
    model_path = "D:/python_opencv/self_driving_car/active_model/lane_navigation_final.h5"
    
    model = load_model(model_path)

    try:
        while True:

            #keyValue = cv2.waitKey(1)
            keyValue = cv2.waitKeyEx(1)


            if keyValue == ord('q') :
                break
            elif keyValue == 2490368 :
                print("go")
                carState = "go"
            elif keyValue == 2621440 :
                print("stop")
                carState = "stop"

            ret, frame = cap.read()

            # Check if the video is ended
            if not ret:
                print("Video playback completed.")
                break

            # 상하좌우 반전
            #frame = cv2.flip(frame, 0)

            frame = cv2.resize(frame, (640, 360))
            #image = DetectLineSlope(frame)[0]
            image = frame

            preprocessed = img_preprocess(image)
            cv2.imshow('pre - Train Model - Ai', preprocessed)
            #cv2.imshow('pre - Train Model - Ai', image)

            X = np.asarray([preprocessed])
            steering_angle = int(model.predict(X)[0])
            print("predict angle:",steering_angle)

            if carState == "go":
                if steering_angle >= 70 and steering_angle <= 110:
                    print("go")
                    #motor_go(speedSet)
                elif steering_angle > 111:
                    print("right")
                    #motor_right(speedSet)
                elif steering_angle < 71:
                    print("left")
                    #motor_left(speedSet)
            elif carState == "stop":
                print("stop")
                #motor_stop()

    except KeyboardInterrupt:
        pass
    

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
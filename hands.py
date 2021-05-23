import cv2
import mediapipe as mp
import csv
import pandas as pd
import tensorflow as tf
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

is_inference = True
is_show_hands = True
classlist = []
filename = 'classlist.txt'
model = None


def save_data(landmark):
    try:
        print("saved dict", landmark)
        df = pd.DataFrame(landmark).T
        df.to_csv('landmark_dataset.csv', mode='a', header=False)
    except Exception as e:
        print("Error exception:", str(e))


def start():
    cur_class_idx = 0
    cap = cv2.VideoCapture('http://192.168.1.18:8080/video')
    # cap = cv2.VideoCapture(0)
    if is_inference:
        model = tf.keras.models.load_model(
            'D:\\Libraries\\Project\\Jupyter Notebook\\TextualEntailment\\best.h5')
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():

            success, image = cap.read()
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = cv2.resize(image, (675, 1200))
            image = image[100: 580, 0: 640]
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            height, width, color = image.shape
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            image = cv2.putText(image, 'CLASS: {}'.format(str(classlist[cur_class_idx])),
                                (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_values = [
                        landmark.x for landmark in hand_landmarks.landmark]
                    y_values = [
                        landmark.y for landmark in hand_landmarks.landmark]

                    x_max = max(x_values)
                    y_max = max(y_values)
                    x_min = min(x_values)
                    y_min = min(y_values)

                    x_center = x_min + ((x_max - x_min) / 2)
                    y_center = y_min + ((y_max - y_min) / 2)

                    key = cv2.waitKey(1)

                    if is_show_hands:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks_data = []
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        # print("Landmark {} {:30} {:30} TO {:30} {:30}".format(i, (landmark.x - x_center), (landmark.y - y_center),(landmark.x - x_center)*width, (landmark.y - y_center)*height))
                        landmarks_data.append(landmark.x - x_center)
                        landmarks_data.append(landmark.y - y_center)

                        norm_landmark = (int((landmark.x-x_center)*width),
                                         int((landmark.y-y_center)*height))

                        # Ini untuk ditampilkan ke window
                        non_norm_landmark = (int(landmark.x*width),
                                             int(landmark.y*height))

                        if is_show_hands:
                            image = cv2.putText(image, '{}, {}'.format(norm_landmark[0], norm_landmark[1]),
                                                (non_norm_landmark),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    label = 'UNKNOWN'

                    if is_inference:
                        prediction = model.predict_classes(
                            np.array([np.array(landmarks_data)]).astype(np.float32))
                        prediction = np.argmax(
                            tf.keras.utils.to_categorical(prediction), axis=1)[0]
                        label = decode_label(prediction)
                    image = cv2.putText(image, '{}'.format(str(label).upper()),
                                        (int(x_center*width),
                                         int(y_center*height)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3, cv2.LINE_AA)
                    # For navigating between classes
                    if key == ord('m'):
                        print("next")
                        if cur_class_idx <= len(classlist)-1:
                            cur_class_idx += 1
                    if key == ord('n'):
                        print("prev")
                        if cur_class_idx > 0:
                            cur_class_idx -= 1
                    # For saving data as dataset
                    if key == ord('c') and not is_inference:
                        landmarks_data.append(classlist[cur_class_idx])
                        save_data(landmarks_data)
                        print("CAPTURING CLASS:", classlist[cur_class_idx])
            # image = cv2.resize(image, (540, 960))
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


def load_ml_model():
    model = tf.keras.models.load_model('model.h5')
    # Check its architecture
    model.summary()

# Single input only


def inference(input):
    try:
        prediction = model.predict_classes(
            np.array([input]).astype(np.float32))
        prediction = np.argmax(
            tf.keras.utils.to_categorical(prediction), axis=1)[0]
        return prediction
    except Exception as e:
        print("Error exception", e)
        return 'ERROR'


def read_classes():
    with open(filename) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            classlist.append(line.strip())
            line = fp.readline()
            cnt += 1
    print("Classlist", classlist)


def decode_label(data):
    return classlist[data]


read_classes()
start()

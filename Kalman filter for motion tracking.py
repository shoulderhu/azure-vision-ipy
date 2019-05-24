import cv2 as cv
import numpy as np
from numpy import uint8, float32, array, zeros


def on_mouse(event, x, y, flags, param):
    global frame, current_measure, current_predict
    last_predict = current_predict
    last_measure = current_measure

    current_measure = array([[float32(x)], [float32(y)]])
    kalman.correct(current_measure)
    current_predict = kalman.predict()

    cv.line(frame, (last_measure[0], last_measure[1]),
            (current_measure[0], current_measure[1]),
            (0, 255, 0), 5)
    cv.line(frame, (last_predict[0], last_predict[1]),
            (current_predict[0], current_predict[1]),
            (0, 0, 255), 5)


def create_frame():
    frame = zeros((720, 1280, 3), uint8)
    frame.fill(255)
    return frame


if __name__ == "__main__":
    winname = "Mouse tracking"
    cv.namedWindow(winname)
    cv.setMouseCallback(winname, on_mouse)

    frame = create_frame()
    current_measure = zeros((2, 1), float32)
    current_predict = zeros((2, 1), float32)

    kalman = cv.KalmanFilter(4, 2)
    kalman.measurementMatrix = array([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], float32)
    kalman.transitionMatrix = array([[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], float32)
    kalman.processNoiseCov = array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]],
                                   float32) * 0.03

    while True:
        cv.imshow(winname, frame)
        key = cv.waitKey(30)

        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("c") or key == ord("C"):
            frame = create_frame()

    cv.destroyAllWindows()


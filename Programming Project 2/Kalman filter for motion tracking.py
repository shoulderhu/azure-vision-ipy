import cv2 as cv
import numpy as np
from numpy import uint8, float32, array, zeros, eye, dot
from numpy.linalg import inv


class KalmanFilter:
    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):

        if F is None or H is None:
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[0]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = eye(self.n) if Q is None else Q
        self.R = eye(self.m) if R is None else R
        self.P = eye(self.n) if P is None else P
        self.x = zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u=0):
        # x(k)=A*x(k-1)+B*u(k)
        self.x = dot(self.F, self.x) + dot(self.B, u)

        # P'(k)=A*P(k-1)*At + Q
        self.P = dot(dot(self.F, self.P), self.F.T) + self.Q
        return dot(self.H, self.x)

    def correct(self, Z):
        # K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
        tmp = dot(dot(self.H, self.P), self.H.T) + self.R
        K = dot(dot(self.P, self.H.T), inv(tmp))

        # x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
        self.x = self.x + dot(K, Z - dot(self.H, self.x))

        # P(k)=(I-K(k)*H)*P'(k)
        self.P = dot(eye(self.n) - dot(K, self.H), self.P)


def on_mouse(event, x, y, flags, param):
    global frame, current_measure
    last_measure = current_measure

    # Kalman Filter prediction and correction
    current_predict = kalman.predict()
    current_predict2 = kalman2.predict()
    current_measure = array([[x], [y]], dtype=float32)
    kalman.correct(current_measure)
    kalman2.correct(current_measure)

    # Draw result on the frame.
    cv.line(frame, (last_measure[0], last_measure[1]),
            (current_measure[0], current_measure[1]),
            GREEN, 3)
    cv.circle(frame, (current_predict[0], current_predict[1]),
              10, RED, 2)
    cv.rectangle(frame, (current_predict2[0] - 5, current_predict2[1] - 5),
                 (current_predict2[0] + 5, current_predict2[1] + 5),
                 BLUE, 2)

    # Print exact location
    print("Mouse location: ", end="")
    print((float(current_measure[0]), float(current_measure[1])))
    print("Kalman Filter predicted the location by OpenCV:")
    print((float(current_predict[0]), float(current_predict[1])))
    print("Kalman Filter predicted the location by hand:")
    print((float(current_predict2[0]), float(current_predict2[1])))
    print("========================================")


def create_frame():
    frame = zeros((720, 1280, 3), uint8)
    frame.fill(255)
    cv.putText(frame, "Mouse Trajectory:", (950, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv.line(frame, (1240, 45), (1260, 45), GREEN, 5)
    cv.putText(frame, "OpenCV Prediction: ", (930, 100),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv.circle(frame, (1250, 95), 10, RED, 2)
    cv.putText(frame, "My Prediction: ", (1008, 150),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv.rectangle(frame, (1250 - 10, 145 - 10), (1250 + 10, 145 + 10), BLUE, 2)
    return frame


if __name__ == "__main__":
    winname = "Kalman Filter Example: Mouse Tracking"
    cv.namedWindow(winname)
    cv.setMouseCallback(winname, on_mouse)

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)

    frame = create_frame()
    current_measure = zeros((2, 1), float32)

    # Parameters for Kalman Filter
    F = array([[1, 0, 1, 0],
               [0, 1, 0, 1],
               [0, 0, 1, 0],
               [0, 0, 0, 1]], float32)
    H = array([[1, 0, 0, 0],
               [0, 1, 0, 0]], float32)
    Q = array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]], float32) * 0.03

    # OpenCV version
    kalman = cv.KalmanFilter(4, 2)
    kalman.measurementMatrix = H
    kalman.transitionMatrix = F
    kalman.processNoiseCov = Q

    # My version
    kalman2 = KalmanFilter(F=F, H=H, Q=Q)

    while True:
        cv.imshow(winname, frame)
        key = cv.waitKey(30)

        if key == ord("q") or key == ord("Q"):
            # Quit
            break
        elif key == ord("c") or key == ord("C"):
            # Clear frame
            frame = create_frame()

    cv.destroyAllWindows()

'''
References:
https://docs.opencv.org/master/dd/d6a/classcv_1_1KalmanFilter.html
https://blog.csdn.net/coming_is_winter/article/details/79048700
'''
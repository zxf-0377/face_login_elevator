import cv2 as cv
import numpy as np

def draw_finger_contours(frame,skin_lower:list=[0, 48, 80],skin_upper:list=[20, 255, 255]):
    hsvim = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower = np.array(skin_lower, dtype = "uint8")
    upper = np.array(skin_upper, dtype = "uint8")
    skinRegionHSV = cv.inRange(hsvim, lower, upper)
    blurred = cv.blur(skinRegionHSV, (2,2))
    ret,thresh = cv.threshold(blurred,0,255,cv.THRESH_BINARY)
    # cv.imshow("thresh", thresh)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours)>0:
        contours = max(contours, key=lambda x: cv.contourArea(x))
    return contours


# cap = cv.VideoCapture(0)
# while True:
#     ret, img = cap.read()
#
#     hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#     lower = np.array([0, 48, 80], dtype = "uint8")
#     upper = np.array([20, 255, 255], dtype = "uint8")
#     skinRegionHSV = cv.inRange(hsvim, lower, upper)
#     blurred = cv.blur(skinRegionHSV, (2,2))
#     ret,thresh = cv.threshold(blurred,0,255,cv.THRESH_BINARY)
#     # cv.imshow("thresh", thresh)
#
#     contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     if len(contours)>0:
#         contours = max(contours, key=lambda x: cv.contourArea(x))
#     cv.drawContours(img, contours, -1, (255,255,0), 2)
#     cv.imshow("contours", img)
#     cv.waitKey(1)
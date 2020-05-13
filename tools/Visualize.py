import cv2
import numpy as np

axis_axis = np.float32([[0,0,0], [0.03,0,0],[0,0.03,0],[0,0,0.03]]).reshape(-1,3)
def draw_axis(img, rvecs, tvecs, mtx, dist):
    imgpts, jac = cv2.projectPoints(axis_axis, rvecs, tvecs, mtx, dist)
    oriPoint = tuple(imgpts[0].ravel())
    x_axis = tuple(imgpts[1].ravel())
    y_axis = tuple(imgpts[2].ravel())
    z_axis = tuple(imgpts[3].ravel())
    
    img = cv2.line(img, oriPoint, x_axis, (255, 0, 0), 5)
    img = cv2.line(img, oriPoint, y_axis, (0, 255, 255), 5)
    img = cv2.line(img, oriPoint, z_axis, (0, 0, 255), 5)

    return img
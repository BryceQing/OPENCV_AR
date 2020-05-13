from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import cv2
def extrinsic2ModelView(matrix):
    """[Get modelview matrix from RVEC and TVEC]

    Arguments:
        RVEC {[vector]} -- [Rotation vector]
        TVEC {[vector]} -- [Translation vector]
    """
    
    Rx = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    
    R  = matrix[:, :3]
    U, S, V = np.linalg.svd(R)
    R = np.dot(U, V)
    R[0, :] = - R[0, :]
    
    #set the translation
    t = matrix[:, 3]
    
    #set 4*4 model view matrix.
    M = np.eye(4)
    M[:3, :3] = np.dot(R, Rx)
    M[:3, 3] = t
    
    return M.T.flatten()


def intrinsic2Project(MTX, width, height, near_plane=0.01, far_plane=100.0):
    """[Get ]

    Arguments:
        MTX {[np.array]} -- [The camera instrinsic matrix that you get from calibrating your chessboard]
        width {[float]} -- [width of viewport]]
        height {[float]} -- [height of viewport]

    Keyword Arguments:
        near_plane {float} -- [near_plane] (default: {0.01})
        far_plane {float} -- [far plane] (default: {100.0})

    Returns:
        [np.array] -- [1 dim array of project matrix]
    """
    P = np.zeros(shape=(4, 4), dtype=np.float32)
    
    fx, fy = MTX[0, 0], MTX[1, 1]
    cx, cy = MTX[0, 2], MTX[1, 2]
    
    
    P[0, 0] = 2 * fx / width
    P[1, 1] = 2 * fy / height
    P[2, 0] = 1 - 2 * cx / width
    P[2, 1] = 2 * cy / height - 1
    P[2, 2] = -(far_plane + near_plane) / (far_plane - near_plane)
    P[2, 3] = -1.0
    P[3, 2] = - (2 * far_plane * near_plane) / (far_plane - near_plane)

    # view_w, view_h = 640, 480
    # fx = MTX[0, 0]
    # fy = MTX[1, 1]
    # fovy = 2 * np.arctan(0.5 * view_h / fy) * view_w / np.pi
    # aspect = (view_w * fy ) / (view_h * fx)
    # gluPerspective(fovy, aspect, near_plane, far_plane)
    # glViewPort(0, 0, view_w, view_h)


    return P.flatten()

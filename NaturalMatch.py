# TODO -> Implement command line arguments (scale, model and object to be projected)
#      -> Refactor and organize code (proper funcition definition and separation, classes, error handling...)

import argparse

import cv2
import numpy as np
import math
import os
from objloader_simple import *

# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 10 


def main():
    """
    This functions loads the target surface image,
    """
    homography = None 
    camera_parameters = np.array([ 
                            [968.40, 0.00, 224.52],
                            [0.00, 964.64, 86.44],
                            [0.00, 0.00, 1.00]
                        ])
    
    
    orb = cv2.ORB_create() # create ORB keypoint detector
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # create BFMatcher object based on hamming distance  
    model = cv2.imread('NaturalMark/refer.png', 0) # load the reference surface that will be searched in the video stream
    kp_model, des_model = orb.detectAndCompute(model, None) # Compute model keypoints and its descriptors 
    obj = OBJ('fox.obj', swapyz=True) # Load 3D model from OBJ file
    cap = cv2.VideoCapture(0)# init video capture

    while True:
        try:
            _, frame = cap.read()# read the current frame
            if not _: return 
            kp_frame, des_frame = orb.detectAndCompute(frame, None)# find and draw the keypoints of the frame
            matches = bf.match(des_model, des_frame)# match frame descriptors with model descriptors
            matches = sorted(matches, key=lambda x: x.distance)# sort them in the order of their distance, lower distance better the match

            if len(matches) > MIN_MATCHES:
                print('find')
                # differenciate between source points and destination points
                src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                # compute Homography
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, mask = 5.0) # compute Homography if enough matches are found
                
                
                
                if homography is not None:
                    try:
                        projection = projection_matrix(camera_parameters, homography)  # obtain 3D projection matrix from homography matrix and camera parameters
                        frame = render(frame, obj, projection, model, False)
                    except:
                        pass
            else:
                pass
            
            res = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:MIN_MATCHES], 0, flags = 2)# draw most matches points.
            cv2.imshow('frame', res)
            
            keyBoard = cv2.waitKey(20)
            if keyBoard  == ord('q'): break
            elif keyBoard  == ord('='): 
                for  i in range(10):
                    temp = cv2.drawMatches(model, kp_model, frame, kp_frame, [matches[i]], 0, flags = 2)# draw most matches points.
                    cv2.waitKey(0)
                    cv2.imshow('frame', temp)
                
                # project_another(camera_parameters, homography)
                print('Debug author result:', projection)
        except:
            pass
            
    cap.release()
    cv2.destroyAllWindows()
    return 0

def render(img, obj, projection, model, color = False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 1
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img


    
    
    
    

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))




# Command line argument parsing
# NOT ALL OF THEM ARE SUPPORTED YET
parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-mk','--model_keypoints', help = 'draw model keypoints', action = 'store_true')
parser.add_argument('-fk','--frame_keypoints', help = 'draw frame keypoints', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')
# TODO jgallostraa -> add support for model specification
#parser.add_argument('-mo','--model', help = 'Specify model to be projected', action = 'store_true')

args = parser.parse_args()

if __name__ == '__main__':
    main()
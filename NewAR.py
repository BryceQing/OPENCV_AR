from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
import cv2.aruco as aruco
from PIL import Image
import numpy as np
import imutils
import sys
import math
 

from objloader import * #Load obj and corresponding material and textures.
from NewMatrixTransfrom import extrinsic2ModelView, intrinsic2Project 



class AR_render:
    
    def __init__(self, camera_matrix, dist_coefs, object_path, model_scale = 0.03):
        
        """[Initialize]
        
        Arguments:
            camera_matrix {[np.array]} -- [your camera intrinsic matrix]
            dist_coefs {[np.array]} -- [your camera difference parameters]
            object_path {[string]} -- [your model path]
            model_scale {[float]} -- [your model scale size]
        """
        # Initialise webcam and start thread
        # self.webcam = cv2.VideoCapture(0)
        self.webcam = cv2.VideoCapture(0)
        self.image_w, self.image_h = map(int, (self.webcam.get(3), self.webcam.get(4)))
        self.initOpengl(self.image_w, self.image_h)
        self.model_scale = model_scale
    
        self.cam_matrix,self.dist_coefs = camera_matrix, dist_coefs
        self.projectMatrix = intrinsic2Project(camera_matrix, self.image_w, self.image_h, 0.01, 100.0)
        self.loadModel(object_path)
        
        # Model translate that you can adjust by key board 'w', 's', 'a', 'd'
        self.translate_x, self.translate_y, self.translate_z = 0, 0, 0
        
        
        # About natural texture
        self.orb = cv2.ORB_create() # create ORB keypoint detector
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # create BFMatcher object based on hamming distance  
        self.imgModel = cv2.imread('NaturalMark/DesignMark.jpg', 0) # load the reference surface that will be searched in the video stream
        self.kp_model, self.des_model = self.orb.detectAndCompute(self.imgModel, None) # Compute model keypoints and its descriptors 
        
        
        
        

    def loadModel(self, object_path):
        
        """[loadModel from object_path]
        
        Arguments:
            object_path {[string]} -- [path of model]
        """
        self.model = OBJ(object_path, swapyz = True)

  
    def initOpengl(self, width, height, pos_x = 500, pos_y = 500, window_name = b'Aruco Demo'):
        
        """[Init opengl configuration]
        
        Arguments:
            width {[int]} -- [width of opengl viewport]
            height {[int]} -- [height of opengl viewport]
        
        Keyword Arguments:
            pos_x {int} -- [X cordinate of viewport] (default: {500})
            pos_y {int} -- [Y cordinate of viewport] (default: {500})
            window_name {bytes} -- [Window name] (default: {b'Aruco Demo'})
        """
        
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(pos_x, pos_y)
     
        
        
        
        self.window_id = glutCreateWindow(window_name)
        glutDisplayFunc(self.draw_scene)
        glutIdleFunc(self.draw_scene)
        
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glShadeModel(GL_SMOOTH)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        
        # Assign texture
        glEnable(GL_TEXTURE_2D)
        self.texture_background = glGenTextures(1)
        
        
        # Set ambient lighting
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
        
        
        
 
    def draw_scene(self):
        """[Opengl render loop]
        """
        _, image = self.webcam.read()# get image from webcam camera.
        # self.draw_background(image)  # draw background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        image = self.draw_objects(image) # draw the 3D objects.
        glutSwapBuffers()
    
        
        # TODO add close button
        key = cv2.waitKey(20)
        
       
        
 
 
 
    # FIXME I think draw background shoule be fixed, but it can work well now.
    def draw_background(self, image):
        """[Draw the background and tranform to opengl format]
        
        Arguments:
            image {[np.array]} -- [frame from your camera]
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Setting background image project_matrix and model_matrix.
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(33.7, 1.3, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
     
        # Convert image to OpenGL texture format
        bg_image = cv2.flip(image, 0)
        bg_image = Image.fromarray(bg_image)     
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)
  
        # Create background texture
        # glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)
        
                
        glTranslatef(0.0,0.0,-10.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 4.0,  3.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-4.0,  3.0, 0.0)
        glEnd()

 
 
 
    def draw_objects(self, image, mark_size = 0.07):
        """[draw models with opengl]
        
        Arguments:
            image {[np.array]} -- [frame from your camera]
        
        Keyword Arguments:
            mark_size {float} -- [aruco mark size: unit is meter] (default: {0.07})
        """
        
        kp_frame, des_frame, = self.orb.detectAndCompute(image, None)
            
        try:    
        
            matches = self.bf.match(self.des_model, des_frame)
            matches = sorted(matches, key = lambda x: x.distance)
            height, width, channels = image.shape
            glClear(GL_DEPTH_BUFFER_BIT)
            
            if len(matches) > 10:
                # print('find')
                image = cv2.drawMatches(self.imgModel, self.kp_model, image, kp_frame, matches[:10], 0, flags = 2)
                src_pts = np.float32([self.kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                # compute Homography
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, mask = 5.0) # compute Homography if enough matches are found
                num, Rs, Ts, Ns = cv2.decomposeHomographyMat(self.cam_matrix, homography)
                # The most important parameter is rvecs and tvecs.
                if homography is not None:
                    # projection matrix.
                    projectMatrix = intrinsic2Project(self.cam_matrix, width, height, 0.01, 100.0)
                    glMatrixMode(GL_PROJECTION)
                    glLoadIdentity()
                    glMultMatrixf(projectMatrix)
                    # intrinsic2Project(self.cam_matrix, width, height, 0.01, 100.0)

                    
                    #model view matrix.
                    cam_pos = self.newTest(self.cam_matrix, homography)
                    
                    glMatrixMode(GL_MODELVIEW)
                    glLoadIdentity()
                    model_matrix = extrinsic2ModelView(cam_pos)
                    glLoadMatrixf(model_matrix)
                    
                    
                    glScaled(self.model_scale, self.model_scale, self.model_scale)
                    glTranslatef(self.translate_x, self.translate_y, self.translate_y)        
                    glCallList(self.model.gl_list)
                
        except:
            pass
        
        cv2.imshow("Frame",image)
    
        
        
    
    def projection_matrix(self, camera_parameters, homography):
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
        # return projection
        
        
        
    
    
    def newTest(self, intrinsic, homography):
        
        
        P = np.hstack((intrinsic, np.dot(intrinsic, np.array([[0],[0],[-1]]))))# camera matrix for query image
        
        P = np.dot(homography, P) # camera matrix for scene image
        
        A = np.dot(np.linalg.inv(intrinsic), P[:,:3])
        A = np.array([A[:, 0], A[:, 1], np.cross(A[:, 0],A[:, 1])]).T

        # camera pose [R|t]
        P[:, :3] = np.dot(intrinsic, A)
        Rt = np.dot(np.linalg.inv(intrinsic), P)
        return Rt
        
    
    def project_another(self, camera_parameters, homography):
        
        norm1 = np.linalg.norm(homography[:, 0])
        norm2 = np.linalg.norm(homography[:, 1])
        tnorm = (norm1 + norm2) / 2.0
        
        H1 = homography[:, 0] / norm1
        H2 = homography[:, 1] / norm2
        
        H3 = np.cross(H1, H2)
        
        T = homography[:, 2] / tnorm
        return np.array([H1, H2, H3, T]).transpose()
    
        
        
    def decHomography(self, A, H):
        H = np.transpose(H)
        h1 = H[0]
        h2 = H[1]
        h3 = H[2]

        Ainv = np.linalg.inv(A)

        L = 1 / np.linalg.norm(np.dot(Ainv, h1))

        r1 = L * np.dot(Ainv, h1)
        r2 = L * np.dot(Ainv, h2)
        r3 = np.cross(r1, r2)

        T = L * np.dot(Ainv, h3)

        R = np.array([[r1], [r2], [r3]])
        R = np.reshape(R, (3, 3))
        U, S, V = np.linalg.svd(R, full_matrices=True)

        U = np.array(U)
        V = np.array(V)
        R = U * V
        

        return (R, T)
        

        
    def run(self):
        # Begin to render
        glutMainLoop()
  

if __name__ == "__main__":
    # The value of cam_matrix and dist_coeff from your calibration by using chessboard.
    cam_matrix = np.array([ 
                    [968.40, 0.00, 224.52],
                    [0.00, 964.64, 86.44],
                    [0.00, 0.00, 1.00]
                ])
    dist_coeff = np.array([-0.50473126, 0.79121745, 0.01319739, 0.0116239, -1.16359485]) 
    ar_instance = AR_render(cam_matrix, dist_coeff, './Models/Box/box.obj', model_scale = 0.03)
    ar_instance.run() 
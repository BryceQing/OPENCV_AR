from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
from PIL import Image
import numpy as np
from objloader import *
from imutils.video import VideoStream
import cv2.aruco as aruco
import imutils

"""
This is file loads and displays the 3d model on OpenGL screen.
"""


def extrinsic2ModelView(RVEC, TVEC):
    """[Get modelview matrix from RVEC and TVEC]
    
    Arguments:
        RVEC {[vector]} -- [Rotation vector]
        TVEC {[vector]} -- [Translation vector]
    """
    R, _ = cv2.Rodrigues(RVEC)
    Rx = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    
    TVEC = TVEC.flatten().reshape((3, 1))
    # print('Debug TVEC', TVEC)
    rv = Rx @ R
    rt = Rx @ TVEC
    
    m = [0.0] * 16
    m[0] = rv[0, 0]
    m[1] = rv[1, 0]
    m[2] = rv[2, 0]
    m[3] = 0.0
    
    m[4] = rv[0, 1]
    m[5] = rv[1, 1]
    m[6] = rv[2, 1]
    m[7] = 0.0
    
    m[8] = rv[0, 2]
    m[9] = rv[1, 2]
    m[10] = rv[2, 2]
    m[11] = 0.0
    
    
    # print('Debug rt', rt.shape)
    
    m[12] = rt[0, 0]
    m[13] = rt[1, 0]
    m[14] = rt[2, 0]
    m[15] = 1.0
    
    return m


def intrinsic2Project(MTX, width, height, near_plane = 0.01, far_plane = 100.0):
    """[Get Projection]

    Arguments:
        width {[int]} -- [The width of viewport]
        height {[int]} -- [The height of viewport]
        MTX {[array]} -- [The internal reference of camera]
    """
    
    
    P = np.zeros(shape = (4, 4), dtype = np.float32)
    fx, fy = MTX[0, 0], MTX[1, 1]
    cx, cy = MTX[0, 2], MTX[1, 2]
    
    
    P[0, 0] = 2 * fx / width
    P[1, 1] = 2 * fy / height
    P[2, 0] = 1 - 2 * cx / width
    P[2, 1] = 2 * cy / height - 1
    P[2, 2] = -( far_plane + near_plane) / (far_plane - near_plane)
    P[2, 3] = -1.0
    P[3, 2] = - ( 2 * far_plane * near_plane) / (far_plane - near_plane)
    
    return P.flatten()



 
 
class OpenGLGlyphs:
  
    # constants
    INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [ 1.0, 1.0, 1.0, 1.0]])
 
    def __init__(self):
        # initialise webcam and start thread
        self.webcam = cv2.VideoCapture(0)
 
        # initialise shapes
        self.wolf = None
        self.file = None
        self.cnt = 1
 

        
        self.texture_background = None

        self.cam_matrix,self.dist_coefs,rvecs,tvecs = self.get_cam_matrix()
        

    def get_cam_matrix(self):
        
        cam_matrix = np.array([ 
                   [968.40, 0.00, 224.52],
                   [0.00, 964.64, 86.44],
                   [0.00, 0.00, 1.00]
                ])
        dist_coeff = np.array([-0.50473126, 0.79121745, 0.01319739, 0.0116239, -1.16359485])
        return cam_matrix, dist_coeff, None, None

  
    def _init_gl(self, Width, Height):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glShadeModel(GL_SMOOTH)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        
        
       
        
        
        # Load 3d object    

        File = 'box.obj'
        self.wolf = OBJ(File,swapyz=True)
        # assign texture
        glEnable(GL_TEXTURE_2D)
        self.texture_background = glGenTextures(1)
        
        #Judge the static object position has been achieved.
        # self.idFind = [False] * 10
        # self.idRvecArr = [None] * 10
        # self.idTVecArr = [None] * 10
        # self.cornersArr = [None] * 10
        
        
        
 
    def _draw_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(33.7, 1.3, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
   
        # get image from webcam
         # get image from camera.
        _, image = self.webcam.read()
 
        # convert image to OpenGL texture format
        bg_image = cv2.flip(image, 0)
        bg_image = Image.fromarray(bg_image)     
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)
  
        # create background texture
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)
         
        # draw background
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glPushMatrix()
        glTranslatef(0.0,0.0,-10.0)
        self._draw_background()
        glPopMatrix()
 
        # handle glyphs
        image = self._handle_glyphs(image)
 
        glutSwapBuffers()
        # cv2.imshow('image', image)
        cv2.waitKey(20)
        
 
    def _handle_glyphs(self, image):

        glClear(GL_DEPTH_BUFFER_BIT)
        
        
        # aruco data
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)      
        parameters =  aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = True

        height, width, channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)

        
        if ids is not None and corners is not None:
            rvecs, tvecs, _= aruco.estimatePoseSingleMarkers(corners, 0.07, self.cam_matrix, self.dist_coefs)
            for i in range(rvecs.shape[0]):
                aruco.drawAxis(image, self.cam_matrix, self.dist_coefs, rvecs[i, :, :], tvecs[i, :, :], 0.03)
            # try:
                
                
                projectMatrix = intrinsic2Project(self.cam_matrix, width, height, 0.01, 100.0)
                glMatrixMode(GL_PROJECTION)
                
                
                reflect = [
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1
                ]
                
                glLoadMatrixf(reflect)
                glMultMatrixf(projectMatrix)


                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                
                
                model_matrix = extrinsic2ModelView(rvecs, tvecs)
                
                glLoadMatrixf(model_matrix)
                # x, y, z = tvecs[0][0]
                # # print(x, y, z)
                # glTranslated(-0.02, -0.13, 0)
                glScaled(0.03, 0.03, 0.03)
                glCallList(self.wolf.gl_list)
                
            # except:
            #     print('error')
            #     pass

        cv2.imshow("cv frame",image)
        
        

    def _draw_background(self):
        # draw background
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 4.0,  3.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-4.0,  3.0, 0.0)
       
        glEnd( )


 
    def main(self):
        # setup and run OpenGL
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(640, 480)
        glutInitWindowPosition(500, 500)
        self.window_id = glutCreateWindow(b"Aruco Demo")
        glutDisplayFunc(self._draw_scene)
        glutIdleFunc(self._draw_scene)
        self._init_gl(640, 480)
        glutMainLoop()
  
# run an instance of OpenGL Glyphs 
openGLGlyphs = OpenGLGlyphs()
openGLGlyphs.main()
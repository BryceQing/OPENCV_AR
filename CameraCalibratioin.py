import numpy as np
import cv2
import glob

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)



width, height = 9, 6 # Size of chessboard is 9 * 6 like ChessBoardSet show, you can change it as you like.
objp = np.zeros((width * height,3), np.float32)
objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./ChessBoardSet/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    w, h = img.shape[:2]
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (width, height ),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        imgName = fname[fname.rindex('\\') + 1 : ]
        # Draw and display the corners
        # You can check the detect result in folder ChessBoardMarkedSet
        img = cv2.drawChessboardCorners(img, (width ,height), corners2,ret)
        cv2.imwrite('./ChessBoardMarkedSet/' + imgName, img)
        

# Get the camera intrinsic matrix and dist_coeff.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist, (w,h), 1, (w,h))

# Write the answer to data1.txt file.
with open('./CameraParameter/data1.txt', 'a') as f:
    f.write('Mtx: \n')
    mat = np.matrix(newcameramtx)
    for line in mat:
        np.savetxt(f, line, fmt = "%.2f")
    f.write('Revcs: \n' + str(rvecs) + '\n')
    f.write('Tvecs: \n' + str(tvecs) + '\n')
    f.write('Dist: \n' + str(dist) + '\n')
            
cv2.destroyAllWindows()
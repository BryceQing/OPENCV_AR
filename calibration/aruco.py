import cv2
import cv2.aruco as aruco
import numpy as np
from imutils.video import VideoStream
import time
import yaml
import imutils
import os

#video source
sc = "http://172.20.10.3:8160/"

#cam_matrix_file name
cam_matrix_file = "chessboard_matrix.yaml"



# Arrays to store object points and image points from all the images.
obj_points = [] # 3d point in real world space
img_points = [] # 2d points in image plane.

h, w = 0,0 #image height and width

term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)

square_size = 0.01 # default in meters
pattern_size = (7, 5)
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size



def get_frame(camera):
	img =  camera.read()
	img = imutils.resize(img,width=640)
	return img

def convert_gray(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img

def get_cam_matrix(file):
	with open(file) as f:
	    loadeddict = yaml.load(f)
	    cam_matrix = np.array(loadeddict.get('camera_matrix'))
	    dist_coeff = np.array(loadeddict.get('dist_coeff'))
	    rvecs = np.array(loadeddict.get('rvecs'))
	    tvecs = np.array(loadeddict.get('tvecs'))
	    return cam_matrix,dist_coeff,rvecs,tvecs

def write_cam_matrix(file,data):
	with open(file, "w") as f:
		yaml.dump(data, f)

def calibrate_chessboard(camera,number_of_data,cam_matrix_file):
	"""

	camera : camera object (imutils VideoStream object)
	number_of_data : number of img points and object point to take until calibration ends
	cam_matrix_file : yaml file to save camera_matrix details

	"""
	error_flag = True
	cnt=1
	while True:
		img = get_frame(camera)
		gray = convert_gray(img)
		found, corners = cv2.findChessboardCorners(gray, pattern_size)
		
		if found==True and len(corners)>0:
			# print(cnt)
			if cnt<=number_of_data:
				cv2.imwrite("chess_board_frames/chess_board___"+str(cnt)+".jpg",img)
				cnt +=1
			else:
				cv2.destroyAllWindows()
				break
			cv2.drawChessboardCorners(img, pattern_size, corners, found)
				
		cv2.imshow('frame',img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

	print(cnt) 
	print(number_of_data)
	for fname in os.listdir("chess_board_frames/"):
		img = cv2.imread("chess_board_frames/"+fname)
		gray = convert_gray(img)
		found, corners = cv2.findChessboardCorners(gray, pattern_size)

		if found:
			cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), term)
			cv2.drawChessboardCorners(img, pattern_size, corners, found)
			img_points.append(corners.reshape(-1, 2))
			obj_points.append(pattern_points)

			
			if len(img_points) ==len(obj_points) and len(obj_points)==number_of_data:
				print("calibration started...")
				cv2.destroyAllWindows()
				h, w = img.shape[:2]
				rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
				newcameramtx, roi=cv2.getOptimalNewCameraMatrix(camera_matrix,dist_coefs,(w,h),1,(w,h))
				# print(camera_matrix,dist_coefs)
				data = {'rms':rms,'camera_matrix': np.asarray(newcameramtx).tolist(), 'dist_coeff': np.asarray(dist_coefs).tolist(), 'rvecs': np.asarray(rvecs).tolist(), 'tvecs': np.asarray(tvecs).tolist()}
				write_cam_matrix(cam_matrix_file,data)
				print("data gathered and written to file!")
				error_flag = False
				break
			else:
				print("No match", len(obj_points))

		cv2.imshow('frame',img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
	if error_flag==True:
		return False
	else:
		return True

def draw_axis_chess(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img



def draw_cube_chess(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-2)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,7)):
    	print(imgpts[i])
    	img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),1)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),1)

    return img


def aruco():

	print("getting data from file")
	cam_matrix,dist_coefs,rvecs,tvecs = get_cam_matrix("camera_matrix_aruco.yaml")

	# aruco data

	aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
	parameters =  aruco.DetectorParameters_create()


	corner_points = [] 
	count =1
	img = vs.read()
	height, width, channels = img.shape
	out = cv2.VideoWriter('outpy1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width,height))
	while True:
		count =1

		img = get_frame(vs)
		gray = convert_gray(img)
		corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

		if ids is not None and corners is not None:
			# obj = aruco.drawDetectedMarkers(img, corners,ids,(255,0,0))

			for cor in corners:
				r,t ,_objpoints = aruco.estimatePoseSingleMarkers(cor,0.5,cam_matrix,dist_coefs)
				# aruco.drawAxis(img,cam_matrix,dist_coefs,r,t,0.2)

		# if len(corners) == len(ids):
		board = aruco.GridBoard_create(6,8,0.05,0.01,aruco_dict)
		corners, ids, rejectedImgPoints,rec_idx = aruco.refineDetectedMarkers(gray,board,corners,ids,rejectedImgPoints)
		ret,rv,tv = aruco.estimatePoseBoard(corners,ids,board,cam_matrix,dist_coefs)
		if ret:
			aruco.drawAxis(img,cam_matrix,dist_coefs,rv,tv,0.2)
			obj_points,imgpoints = aruco.getBoardObjectAndImagePoints(board,corners,ids)

			# imgpts = np.int32(imgpoints).reshape(-1,2)
			# # draw ground floor in green
			# img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),3)


		

		# out.write(img)
		cv2.imshow("Markers",img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	out.release()
	cv2.destroyAllWindows()



def get_chessboard_frames(camera,number_of_data):

	"""
	
	camera : camera object (imutils VideoStream object)
	number_of_data : number of img points and object point to take until calibration ends

	"""

	error_flag = False
	cnt=1
	while True:
		img = get_frame(camera)
		gray = convert_gray(img)
		found, corners = cv2.findChessboardCorners(gray, pattern_size)
		
		if found==True and len(corners)>0:
			# print(cnt)
			if cnt<=number_of_data:
				if not os.path.exists("chess_board_frames"):
					os.mkdir("chess_board_frames")
				cv2.imwrite("chess_board_frames/chess_board___"+str(cnt)+".jpg",img)
				cnt +=1
			else:
				error_flag = True
				cv2.destroyAllWindows()
				break
			cv2.drawChessboardCorners(img, pattern_size, corners, found)
				
		cv2.imshow('frame',img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

	print(cnt) 
	print("\n\n"+str(number_of_data)+" data written in chess_board_frames directory")
	return error_flag


def start_calibration_chessboard(number_of_data):

	for fname in os.listdir("chess_board_frames/"):
		img = cv2.imread("chess_board_frames/"+fname)
		gray = convert_gray(img)
		found, corners = cv2.findChessboardCorners(gray, pattern_size)

		if found:
			cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), term)
			cv2.drawChessboardCorners(img, pattern_size, corners, found)
			img_points.append(corners.reshape(-1, 2))
			obj_points.append(pattern_points)

			
			if len(img_points) ==len(obj_points) and len(obj_points)==number_of_data:
				print("calibration started...")
				cv2.destroyAllWindows()
				h, w = img.shape[:2]
				rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
				ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1],None,None)
				newcameramtx, roi=cv2.getOptimalNewCameraMatrix(camera_matrix,dist_coefs,(w,h),1,(w,h))
				dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
				x,y,w,h = roi
				dst = dst[y:y+h, x:x+w]
				cv2.imwrite('calib_res/calibresult.png',dst)
				data = {'rms':rms,'camera_matrix': np.asarray(newcameramtx).tolist(), 'dist_coeff': np.asarray(dist_coefs).tolist(), 'rvecs': np.asarray(rvecs).tolist(), 'tvecs': np.asarray(tvecs).tolist()}
				write_cam_matrix(cam_matrix_file,data)
				print(data)
				print("data gathered and written to file!")
				error_flag = False
				break
			else:
				print("No match", len(obj_points))

		cv2.imshow('frame',img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
	if error_flag==True:
		return False
	else:
		return True


def chessboard_3d(camera,cam_matrix_file):
	# calibrate_chessboard(vs,50,"chessboard_matrix.yaml")

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	objp = np.zeros((6*7,3), np.float32)
	objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

	# axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3) # axis
	axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ]) #cube
	cam_matrix,dist_coeff,rvecs,tvecs = get_cam_matrix("chessboard_matrix.yaml")

	img_points = []
	obj_points = []
	print("main started")
	cnt=0
	while True:
		img = camera.read()
		gray = convert_gray(img)
		ret, corners = cv2.findChessboardCorners(gray, pattern_size)

		if ret:
			corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
			#cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
			if corners2 is not None:
				ret,rvecs, tvecs, inliers = cv2.solvePnPRansac(pattern_points, corners2, cam_matrix, dist_coeff)
				#ret,rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, cam_matrix, dist_coeff)
				if ret:
					# print(obj_points)
					# project 3D points to image plane
					imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cam_matrix, dist_coeff)
					try:
						draw_cube_chess(img,corners2,imgpts)
						#cv2.imwrite("output/Cube_drawn_"+str(cnt)+".jpg",img)
						cnt +=1		
					except:
						print("some error")
			else:
				print("No")
				print(obj_points)

		cv2.imshow('img',img)
		k = cv2.waitKey(1) & 0xff
		if k == 's':
			cv2.imwrite("detteed"+'.png', img)

	cv2.destroyAllWindows()



def main():
	vs = VideoStream(src=sc).start()
	time.sleep(1)
	cam_matrix,dist_coeff,rvecs,tvecs = get_cam_matrix(cam_matrix_file)

	"""
	flag = calibrate_chessboard(vs,20,cam_matrix_file)
	if flag:
		main()
	else:
		print("calibration error")
	"""

	#step:1 (run step 1 if calibration is not done.
	"""
	ret = get_chessboard_frames(vs,20)
	if ret:
		flag = start_calibration_chessboard(20)
		if flag:
			print("\ncalibaration done")
	"""

	#  step:3
	chessboard_3d(vs,cam_matrix_file)

main()



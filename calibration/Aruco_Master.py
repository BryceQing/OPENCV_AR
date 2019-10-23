import cv2.aruco as aruco
from Imutils_Master import Imutils_Master
import imutils
import cv2
import yaml
import numpy as np
import os

class Aruco_Master(object):

	term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
	square_size = 1 # default in meters
	pattern_size = (6, 5)
	pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
	pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
	pattern_points *= square_size

	aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
	parameters =  aruco.DetectorParameters_create()
	obj_points = [] # 3d point in real world space
	img_points = [] # 2d points in image plane.
	h, w = 0,0 #image height and width
	MarkerX  = 0
	MarkerY = 0
	marker_length = 0
	space_bet_marker = 0
	board = None
	rvecs= None
	tvecs= None
	outputfile = None
	

	def __init__(self,MarkerX_cols,MarkerY_rows,marker_length=0.05,space_bet_marker=0.001,dictionary=None,outputfile="camera_matrix_aruco.yaml"):
		self.MarkerX = MarkerX_cols
		self.MarkerY = MarkerY_rows
		self.marker_length = marker_length
		self.outputfile = outputfile
		self.space_bet_marker = space_bet_marker
		if dictionary is not None:
			self.aruco_dict = dictionary

	# def random_things(self):
	# 	board = aruco.GridBoard_create(self.MarkerX,self.MarkerY,0.05,0.01,self.aruco_dict)
	# 	cam_matrix,dist_coeff,rvecs,tvecs= self.read_cam_matrix(self.outputfile)
	# 	im = Imutils_Master()
	# 	total_makers_to_look = self.MarkerX * self.MarkerY
	# 	axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
	# 	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
	# 	objp = np.zeros((6*7,3), np.float32)
	# 	objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
	# 	while True:
	# 		img = im.getframe()
	# 		gray = im.gray(img)
	# 		corners,ids,rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
	# 		corners1,ids1,rejectedImgPoints1,rec_index = aruco.refineDetectedMarkers(gray,board, corners, ids,rejectedImgPoints)
	# 		getcount = self.get_marker_count(ids1)

	# 		if ids1 is not None and getcount==total_makers_to_look:
	# 				ret,rv,tv = aruco.estimatePoseBoard(corners1,ids1,board,cam_matrix,dist_coeff)
	# 				if ret>0:
	# 					# aruco.drawDetectedMarkers(img, corners1,ids1,(0,255,0))
	# 					obj_points,img_points = aruco.getBoardObjectAndImagePoints(board,corners1,ids1)
	# 					if img_points is not None:
	# 						# corners2 = cv2.cornerSubPix(gray,np.array(corners1),(6,6),(-1,-1),criteria)
	# 						ret,rvecs, tvecs, inliers = cv2.solvePnPRansac(obj_points,img_points, cam_matrix, dist_coeff)
	# 						imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cam_matrix, dist_coeff)
	# 						# img = self.draw(img,corners1,img_points)
	# 						aruco.drawAxis(img,cam_matrix,dist_coeff,rv,tv,0.2)
		
	# 		cv2.imshow("frame",img)
	# 		if cv2.waitKey(1) & 0xFF == ord('q'):
	# 			break

	def draw_axis_on_board(self):
		board = aruco.GridBoard_create(self.MarkerX,self.MarkerY,0.5,0.01,self.aruco_dict)
		cam_matrix,dist_coeff,rvecs,tvecs= self.read_cam_matrix(self.outputfile)
		im = Imutils_Master()
		while True:
			img = im.getframe()
			gray = im.gray(img)
			corners,ids,rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
			corners1,ids1,rejectedImgPoints1,rec_index = aruco.refineDetectedMarkers(gray,board, corners, ids,rejectedImgPoints)
			getcount = self.get_marker_count(ids1)
			if getcount>0:
				ret,rv,tv = aruco.estimatePoseBoard(corners1,ids1,board,cam_matrix,dist_coeff)
				if ret==2:
					# print(ret,rv,tv)
					# obj_points,imgpoints = aruco.getBoardObjectAndImagePoints(board,corners1,ids1)
					# ret,rv1,tv1,inline = cv2.solvePnPRansac(obj_points, imgpoints, cam_matrix, dist_coeff)
					# ret,rv1,tv1 = cv2.solvePnP(obj_points, imgpoints, cam_matrix, dist_coeff)
					aruco.drawAxis(img,cam_matrix,dist_coeff,rv,tv,0.3)
					# imgpts = np.int32(imgpoints[1]).reshape(-1,2)
					# # draw ground floor in green
					# img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
				
				# for cor in corners1:
				# 	r,t ,_objpoints = aruco.estimatePoseSingleMarkers(cor,0.5,cam_matrix,dist_coeff)
				# 	# aruco.drawDetectedMarkers(img, corners,ids,(0,255,0))
				# 	if r is not None:
				# 		aruco.drawAxis(img,cam_matrix,dist_coeff,r,t,0.3)


			
			cv2.imshow("frame",img)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

	def calibrate_camera_aruco(self,samples_dir="frames",outputfile="camera_matrix_aruco.yaml"):
		self.outputfile = outputfile
		board = aruco.GridBoard_create(self.MarkerX,self.MarkerY,0.05,0.01,self.aruco_dict)
		cam_matrix,dist_coeff,rvecs,tvecs= self.read_cam_matrix(outputfile)
		print("dist_coeff original")
		print(dist_coeff)
		print("cam_matrix original")
		print(cam_matrix)
		im = Imutils_Master()
		all_img = []
		all_obj = []
		h,w = 0,0
		file_count  =0
		total_makers_to_look = self.MarkerX * self.MarkerY
		for fname in os.listdir(samples_dir):
			file_count +=1
			img = cv2.imread(samples_dir+"/"+fname)
			h, w = img.shape[:2]
			gray = im.gray(img)
			corners,ids,rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
			corners1,ids1,rejectedImgPoints1,rec_index = aruco.refineDetectedMarkers(gray,board, corners, ids,rejectedImgPoints)
			getcount = self.get_marker_count(ids1)

			if ids1 is not None and getcount==total_makers_to_look:
				ret,rv,tv = aruco.estimatePoseBoard(corners1,ids1,board,cam_matrix,dist_coeff,rvecs,tvecs)
				if ret>0:
					aruco.drawDetectedMarkers(img, corners,ids,(0,0,255))
					obj_points,img_points = aruco.getBoardObjectAndImagePoints(board,corners1,ids1)
					all_obj.append(obj_points)
					all_img.append(img_points)
				else:
					print("not able to estimate board")
			cv2.imshow("frame",img)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		cv2.destroyAllWindows()
		print("calibrating starts... may take while")
		
		if len(all_obj)==len(all_img):
			rms, cam_matrix, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(all_obj, all_img, (w, h), None, None)
			if rms:
				data = {'camera_matrix': np.asarray(cam_matrix).tolist(),
				 		'dist_coeff': np.asarray(dist_coeff).tolist(),
				  		'rvecs': np.asarray(rvecs).tolist(), 
				  		'tvecs': np.asarray(tvecs).tolist()}
				flag = self.write_cam_matrix(outputfile,data)
				if flag:
					print("camera details is written to file")
					cam_matrix,dist_coeff,rvecs,tvecs= self.read_cam_matrix(outputfile)
					print("new cam_matrix")
					print(cam_matrix)
					print("new dist_coeff")
					print(dist_coeff)
				else:
					print("error writing camera details to file")
			
		else:
			print("Number of object points is not equal to the number of samples taken")
			print(len(all_obj))
			print(len(all_img))
				

	def calibrate_camera_aruco_init(self,output_samples_dir,number_of_samples_to_take):
		im = Imutils_Master("http://172.20.10.3:8160/") #change your video source here.
		img = im.getframe()
		h,w,channel = img.shape
		cnt=0
		total_makers_to_look = self.MarkerX * self.MarkerY
		if not os.path.exists(output_samples_dir):
			os.makedirs(output_samples_dir)
		while True:
			img = im.getframe()
			gray = im.gray(img)
			corners,ids,rejectedImgPoints = self.detect_markers(gray)
			if corners is not None and ids is not None and len(corners)!=0:
				count = self.get_marker_count(ids)
				if count==total_makers_to_look:
					cnt +=1
					if not os.path.exists(output_samples_dir):
						os.mkdir(output_samples_dir)
					cv2.imwrite(output_samples_dir+"/aruco__"+str(count)+"__"+str(cnt)+".jpg",img)
					aruco.drawDetectedMarkers(img, corners,ids,(255,0,0))
			if cnt>=number_of_samples_to_take:
				print(str(number_of_samples_to_take)+" Samples Collected")
				break
			
			cv2.imshow("frame",img)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		
	def detect_markers(self,gray):
		corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
		aruco.drawDetectedMarkers(gray, corners,ids,(255,0,0))
		return corners,ids,rejectedImgPoints

	def get_marker_count(self,corners):
		if corners is not None:
			cnt =0 
			for con in corners:
				cnt +=1
			return cnt
		else:
			return 0

	def write_cam_matrix(self,file=None,Data=None):
		if file is not None:
			with open(file, "w") as f:
				yaml.dump(Data, f)
				return True
		else:
			print("Please Specify Cam Matrix yaml file name to write Data")

	def read_cam_matrix(self,file=None):
		if file is not None:
			with open(file) as f:
			    loadeddict = yaml.load(f)
			    cam_matrix = np.array(loadeddict.get('camera_matrix'))
			    dist_coeff = np.array(loadeddict.get('dist_coeff'))
			    rvecs = np.array(loadeddict.get('rvecs'))
			    tvecs = np.array(loadeddict.get('tvecs'))
			    return cam_matrix,dist_coeff,rvecs,tvecs
		else:
			print("Please Specify Cam Matrix yaml file name to read Data")

	def draw(self,img, corners, imgpts):
		imgpts = np.int32(imgpts).reshape(-1,2)
		# draw ground floor in green
		img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
		return img





	







		
from Aruco_Master import Aruco_Master


output_sample_dir = "frames" # frames will be saved in this directory
num_of_smaples_to_take = 20 # 20 samples will be taken for calibration
output_file = "camera_matrix_aruco.yaml" #camera matrix will be written to this file


# specify rows and columns of your aruco marker sheet so the algorithm will look for n= cols * rows 
# 48 markers in this case
Marker_rows = 8
Marker_cols = 6

Am = Aruco_Master(Marker_cols,Marker_rows) # initialize Object

print("Collecting samples...")
Am.calibrate_camera_aruco_init(output_sample_dir,num_of_smaples_to_take) # take frames to calibrate leter
print("starting calibration...")
Am.calibrate_camera_aruco(output_sample_dir,output_file) # actual calibration 

# Am.random_things()
# Am.draw_axis_on_board()



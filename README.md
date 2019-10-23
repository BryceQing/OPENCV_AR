### This is Aruco code for Python using python bindings of opencv.
```markdown
- Requirements

1. python 3.x
2. opencv 3.x 
3. opencv_contrib (for aruco)


- Packages requirements for python & opencv
install it using pip commands.

1. imutils
2. yaml
3. pygame
4. openGL

```

### I have used anaconda 4 with python version 3.x in windows 10.
```markdown

1. main.py

This file contains the main code which loads the 3d object in program, 
also initializes your camera and finds aruco marker and places the 3d object above the aruco marker.

2. objloader.py

This file is provided by pygame
refer to www.pygame.org/wiki/OBJFileLoader

3. Obj and Mtl files.

Obj file is generally created by blender or any 3d object creator program. (gaming softwares)
Mtl is material for the Obj file. (must require or your object will be rendered as black)

# How to use this.

Make sure you calibrate your camera first from calibration directory. 
(main.py from calibration includes comments to understand)
the generated .yaml file should be used in main.py 

# Calibration

In Aruco_python calibration can be done two ways.

1.Using chess board.
2.Using aruco board.

# For calibrating using aruco (refer calibration/main.py)

- [x] Set your video source in calibration/main.py file.
- [x] Set Camera matrix file name.


Once it says calibration done and data written in file.
Copy the calibrated yaml file to the main directory and replace the older file (must).

# For calibrating using chessboard (refer calibration/aruco.py)

- [x] Set your video source in aruco.py file.
- [x] Set Camera matrix file name.

Rename the yaml file to the same name as main directory yaml file (must).

Now just run the root main.py

Make sure you put all files in one directory.

1. 3dObj file & Mtl file
2. camera_matrix.yaml

```

- How to modify and use other 3d objects.

### Requirements
```markdown
You need to have following

1. Brain (must require)
2. python opengl (an introduction to opengl, and how it works. )
3. blender (to modify and to create new objects as per your requirements.)
4. unity 3d (optional)
5. Aruco marker printed (Refer calibaration/Marker Boards/)


visit https://free3d.com/3d-models/obj-file

Download any .obj file that you might require.
Make sure you download obj file that have mtl file with it or only obj file will render black obj on your screen.
```
Bugs

1. Light is little low on opengl window as compared to opencv window. (make sure you work in environment which have enough light)

Anything you find let me know. :)


- Licence

I do not own any of the obj files, please read the readme file model/zip/sinband directory.

Refernces

1. https://www.uco.es/investiga/grupos/ava/node/26
2. https://rdmilligan.wordpress.com/2015/10/15/augmented-reality-using-opencv-opengl-and-blender/
3. http://jevois.org/moddoc/DemoArUco/modinfo.html



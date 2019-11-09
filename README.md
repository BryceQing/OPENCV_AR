# Progress about this program.
- ## Configration
   - python3.6 + windows10
   - Install external lib
        - First, you should install opencv and else libs in current directory
       ```
       pip install -r requirements.txt
       ```
        - Second, you should install opengl and pygame(which used to load textures) in your python environment. There I have provided the corresponding libs in [ExternalConfig]() folder, you can copy them to your python env and directly install e.g
        ```
        python -m pip --user pygame-1.9.6-cp37-cp37m-win_amd64.whl
        ```
- ## Get camera information
  ### In the program, we need your camera information including intrinsic matrix and distortion coefficient and we use chessboard calibration to get it.
  - First, you should download a chessboard image, there I have provided for you [ChessBoard](chessboard.png). After that, use the camera to capture at least 20 photos and save them to [ChessBoardSet](ChessBoardSet/) as I did.
  - Second, execute [CameraCalibratioin.py](CameraCalibratioin.py) and you can find camera information in [CameraParameter/data1.txt](CameraParameter/data1.txt)

- ## Run the program
  - First, copy the camera intrinsic matrix and distortion coefficient to replace corresponding paramaters [AR_entrance.py](AR_entrance.py)
  - Second, run the [AR_entrance.py](AR_entrance.py) and you can change models which are saved in [Models](Models/) directory.
  - Third, you can press **'+'** or **'-'** to scale the model size as video shows.
  - Note: you can generating the aruco mark by your self on [aruco generator](http://chev.me/arucogen/)

  
- ## Contribute
  If you are interested in opencv, opengl and AR welcome to become a contributor, I will respond you as soon as possible.

- ## Contact
  - You can commit the issue or send an email to bryceqing@zju.edu.cn 
  - **If you like it please give me a star and fork it :) :)**
  
- ## Demo Video
  - [Youtube Video](https://www.youtube.com/watch?v=WT740R5RAMo)
  - [iqiyi Video](http://www.iqiyi.com/w_19sb1h12f1.html)

## References
  > https://github.com/ajaymin28/-Aruco_python  
  > https://github.com/ygx2011/Marker_AR  
  > https://github.com/GeekLiB/AR-BXT-AR4Python  
  > https://github.com/RaubCamaioni/OpenCV_Position
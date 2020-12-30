"""
Author  : Joyce
Date    : 2020-12-26
"""
from UI.hw2_ui import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

class PyMainWindow(QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        super(PyMainWindow, self).__init__()
        self.setupUi(self)
        # === push button clicked action === 
        self.pushButton_3.clicked.connect(self.find_corner)
        self.pushButton_4.clicked.connect(self.find_intrinsic)
        self.pushButton_6.clicked.connect(self.find_distortion)
        self.pushButton_5.clicked.connect(self.find_extrinsic)
        self.pushButton_7.clicked.connect(self.augmented_reality)
        self.pushButton_8.clicked.connect(self.stereo_disparity)

        # === combo box change action ===
        self.comboBox.currentIndexChanged.connect(self.select_image)
        self.show()

        # === Q2 ===
        self.q2_3_img_idx = 0

    """
    ref: https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
    """ 
    def find_corner(self):
        # print('Find corner')
        opencv = OpenCv()
        opencv.Q2()
        opencv.Q2_1()

    def find_intrinsic(self):
        # print('Find intrinsic')
        opencv = OpenCv()
        opencv.Q2()
        opencv.Q2_2()

    def find_distortion(self):
        # print('Find distortion')
        opencv = OpenCv()
        opencv.Q2()
        opencv.Q2_4()
    
    def find_extrinsic(self):
        # print('Find extrinsic')
        opencv = OpenCv()
        opencv.Q2()
        opencv.Q2_3(self.q2_3_img_idx)

    def select_image(self, text):
        self.q2_3_img_idx = int(text)

    def augmented_reality(self):
        # print('Augmented reality')
        opencv = OpenCv()
        opencv.Q3()

    def stereo_disparity(self):
        # print('Stereo disparity')
        opencv = OpenCv()
        opencv.Q4()

class OpenCv(object):
    def __init__(self):
        super(OpenCv, self).__init__()
        # === Q2 ===
        self.corner_img = np.empty(15, dtype=object)    # corner images
        # Arrays to store object points and image point from all the image
        self.objpoints = []                             # 3d point in real world space
        self.imgpoints = []                             # 2d points in image plane
        self.gray_img = np.empty(15, dtype=object)          # gray images
        # ==========

    def Q2(self):
        print('calculate...')
        # Prepare object points, like (0, 0, 0), (1, 0, 0), ..., (7, 10, 0)
        objp = np.zeros((8*11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        
        for f_idx in range(1, 16):
            # Read image and converty to gray image
            img = cv2.imread('./Q2_Image/'+str(f_idx)+'.bmp')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.gray_img[f_idx-1] = gray
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
                # Draw and display the corner
                cv2.drawChessboardCorners(img, (11, 8), corners, ret)
                # save corner images
                self.corner_img[f_idx-1] = img
            
    def Q2_1(self):
        for f_idx in range(1, 16):
            cv2.namedWindow('2.1 Find Corners: '+str(f_idx)+'.bmp', cv2.WINDOW_NORMAL)
            cv2.imshow('2.1 Find Corners: '+str(f_idx)+'.bmp', self.corner_img[f_idx-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def Q2_2(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray_img[14].shape[::-1], None, None)
        print(mtx)

    def Q2_3(self, img_idx):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray_img[img_idx].shape[::-1], None, None)
        R = cv2.Rodrigues(rvecs[img_idx])[0]
        t = tvecs[img_idx]
        extrinsic_matrix = np.hstack([R, t])
        print(extrinsic_matrix)

    def Q2_4(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray_img[0].shape[::-1], None, None)
        print(dist)

    def Q3(self):
        # Prepare object points, like (0, 0, 0), (1, 0, 0), ..., (7, 10, 0)
        objp = np.zeros((8*11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []                             # 3d point in real world space
        imgpoints = []                             # 2d points in image plane

        # Calculate camera calibration coefficients
        for f_idx in range(1, 6):
            # Read image and converty to gray image
            img = cv2.imread('./Q3_Image/'+str(f_idx)+'.bmp')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        cv2.namedWindow('Augmented Reality', cv2.WINDOW_NORMAL)
        # Draw a tetrahedron on chessboard
        for f_idx in range(1, 6):
            # Read image and converty to gray image
            img = cv2.imread('./Q3_Image/'+str(f_idx)+'.bmp')
            red = [0, 0, 255]   # BGR
            # tetrahefron cooridinates
            tetrahefron_coor = np.array([(3, 3, -3),
                                         (3, 5, 0),
                                         (5, 1, 0),
                                         (1, 1, 0)], dtype=np.float32)
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(tetrahefron_coor, rvecs[f_idx-1], tvecs[f_idx-1], mtx, dist)
            imgpts = np.int32(imgpts).reshape(-1, 2)
            # draw the tetrahefron 
            cv2.line(img, tuple(imgpts[0]), tuple(imgpts[1]), red, 10)
            cv2.line(img, tuple(imgpts[0]), tuple(imgpts[2]), red, 10)
            cv2.line(img, tuple(imgpts[0]), tuple(imgpts[3]), red, 10)
            
            cv2.line(img, tuple(imgpts[1]), tuple(imgpts[2]), red, 10)
            cv2.line(img, tuple(imgpts[2]), tuple(imgpts[3]), red, 10)
            cv2.line(img, tuple(imgpts[3]), tuple(imgpts[1]), red, 10)

            cv2.imshow('Augmented Reality', img)
            cv2.waitKey(500)
        cv2.destroyAllWindows()

    


    def Q4(self):
        imgL = cv2.imread('./Q4_Image/imgL.png', 0)
        imgR = cv2.imread('./Q4_Image/imgR.png', 0)

        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(imgL, imgR)
        disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.namedWindow('Q4', cv2.WINDOW_NORMAL)

        def select_point(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                gray = disparity.copy()
                output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                ret_x = output.shape[1]-550
                ret_y = output.shape[0]-150
                cv2.circle(output, (x, y), 10, (255, 0, 0), -1)
                # === Write Text ===
                # rectangle
                cv2.rectangle(output, (ret_x, ret_y), (ret_x + 550, ret_y + 150), (255, 255, 255), -1)
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (ret_x, ret_y + 50) 
                fontScale = 1.7
                fontColor = (0, 0, 0)
                lineType = 4
                # putText
                cv2.putText(
                    output,
                    f'Disparity: {gray[y][x]} pixels',
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )
                d = 178 * 2826 / (gray[y][x] + 123)
                bottomLeftCornerOfText = (ret_x, ret_y + 130) 
                cv2.putText(
                    output,
                    f'Depth: {int(d)} mm',
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )
                cv2.imshow('Q4', output)
        
        # cv2.imshow('Q4', disparity)
        cv2.setMouseCallback('Q4', select_point)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = PyMainWindow()
    ui.setupUi(window)
    ui.show()
    sys.exit(app.exec_())
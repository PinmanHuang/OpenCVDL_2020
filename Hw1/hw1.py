"""
Author  : Joyce
Data    : 2020-10-27
"""
from UI.hw1_ui import Ui_MainWindow
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
        # Q1
        self.pushButton.clicked.connect(self.load_image)
        self.pushButton_2.clicked.connect(self.color_seperation)
        self.pushButton_3.clicked.connect(self.image_flipping)
        self.pushButton_4.clicked.connect(self.blending)
        # Q2
        self.pushButton_5.clicked.connect(self.median_filter)
        self.pushButton_6.clicked.connect(self.gaussian_blur)
        self.pushButton_7.clicked.connect(self.bilateral_filter)
        # Q3
        self.pushButton_8.clicked.connect(self.gaussian_blur_2)
        self.pushButton_9.clicked.connect(self.sobel_x)
        self.pushButton_10.clicked.connect(self.sobel_y)
        self.pushButton_11.clicked.connect(self.magnitude)
        # Q4
        self.rotation = ""
        self.scaling = ""
        self.tx = ""
        self.ty = ""
        self.lineEdit.textChanged.connect(self.rotation_changed)
        self.lineEdit_2.textChanged.connect(self.scaling_changed)
        self.lineEdit_3.textChanged.connect(self.tx_changed)
        self.lineEdit_4.textChanged.connect(self.ty_changed)
        self.pushButton_12.clicked.connect(self.transformation)

    
    # === Q1 ===
    def load_image(self):
        print('Load image')
        opencv = OpenCv()
        opencv.Q1_1()

    def color_seperation(self):
        print('Color seperation')
        opencv = OpenCv()
        opencv.Q1_2()
    
    def image_flipping(self):
        print('Image flipping')
        opencv = OpenCv()
        opencv.Q1_3()
    
    def blending(self):
        print('Blending')
        opencv = OpenCv()
        opencv.Q1_4()

    # === Q2 ===
    def median_filter(self):
        print('Median filter')
        opencv = OpenCv()
        opencv.Q2_1()

    def gaussian_blur(self):
        print('Gaussian blur')
        opencv = OpenCv()
        opencv.Q2_2()
    
    def bilateral_filter(self):
        print('Bilateral filter')
        opencv = OpenCv()
        opencv.Q2_3()

    # === Q3 ===
    def gaussian_blur_2(self):
        print('Gaussian blur')
        opencv = OpenCv()
        opencv.Q3_1()

    def sobel_x(self):
        print('Sobel x')
        opencv = OpenCv()
        opencv.Q3_2()
    
    def sobel_y(self):
        print('Sobel x')
        opencv = OpenCv()
        opencv.Q3_3()
    
    def magnitude(self):
        print('Magnitude')
        opencv = OpenCv()
        opencv.Q3_4()

    # === Q4 ===
    def rotation_changed(self, text):
        self.rotation = text

    def scaling_changed(self, text):
        self.scaling = text

    def tx_changed(self, text):
        self.tx = text
    
    def ty_changed(self, text):
        self.ty = text

    def transformation(self):
        print('transformation')
        # print('rotation: {}\tscaling: {}\ttx: {}\tty: {}'.format(
            # str(self.rotation), str(self.scaling), str(self.tx), str(self.ty)))
        opencv = OpenCv()
        opencv.Q4(self.rotation, self.scaling, self.tx, self.ty)

class OpenCv(object):
    def __init__(self):
        super(OpenCv, self).__init__()

    def Q1_1(self):
        img = cv2.imread('./Q1_Image/Uncle_Roger.jpg')
        cv2.namedWindow('1', cv2.WINDOW_NORMAL)
        cv2.imshow('1', img)
        print('Height = ', img.shape[0])
        print('Width = ', img.shape[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def Q1_2(self):
        img = cv2.imread('./Q1_Image/Flower.jpg')
        # extract channel
        b, g, r = cv2.split(img)

        # create empty image with the same shape as origin image
        b_img = np.zeros(img.shape, dtype='uint8')
        g_img = np.zeros(img.shape, dtype='uint8')
        r_img = np.zeros(img.shape, dtype='uint8')

        # assign channel to empty image
        b_img[:,:,0] = b
        g_img[:,:,1] = g
        r_img[:,:,2] = r

        cv2.namedWindow('blue', cv2.WINDOW_NORMAL)
        cv2.namedWindow('green', cv2.WINDOW_NORMAL)
        cv2.namedWindow('red', cv2.WINDOW_NORMAL)
        cv2.imshow('blue', b_img)
        cv2.imshow('green', g_img)
        cv2.imshow('red', r_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Q1_3(self):
        img = cv2.imread('./Q1_Image/Uncle_Roger.jpg')
        cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Original Image', img)
        img_flip = cv2.flip(img, 1)
        cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        cv2.imshow('Result', img_flip)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Q1_4(self):
        def do_nothing(x):
            pass
        img = cv2.imread('./Q1_Image/Uncle_Roger.jpg')
        img_flip = cv2.flip(img, 1)
        cv2.namedWindow('BLENDING', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('BLEND', 'BLENDING', 0, 255, do_nothing)
        print('please press ESC to close window...')
        while(True):
            alpha = cv2.getTrackbarPos('BLEND', 'BLENDING')/255
            beta = 1.0 - alpha
            blend_img = cv2.addWeighted(img, alpha, img_flip, beta, 0.0)
            cv2.imshow('BLENDING', blend_img)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

    def Q2_1(self):
        # ref: https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9
        img = cv2.imread('./Q2_Image/Cat.png')
        median_img = cv2.medianBlur(img, 7)
        cv2.namedWindow('median', cv2.WINDOW_NORMAL)
        cv2.imshow('median', median_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Q2_2(self):
        # ref: https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
        img = cv2.imread('./Q2_Image/Cat.png')
        gaussian_img = cv2.GaussianBlur(img, (3, 3), 0)
        cv2.namedWindow('Gaussian', cv2.WINDOW_NORMAL)
        cv2.imshow('Gaussian', gaussian_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Q2_3(self):
        # ref: https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
        img = cv2.imread('./Q2_Image/Cat.png')
        bilateral_img = cv2.bilateralFilter(img, 9, 90, 90)
        cv2.namedWindow('Bilateral', cv2.WINDOW_NORMAL)
        cv2.imshow('Bilateral', bilateral_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def convolve2d(self, image, kernel, padding=0, strides=1):
        # ref: https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
        # cross Correlation
        kernel = np.flipud(np.fliplr(kernel))

        # Gather Shapes of Kernel + Image + Padding
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]
        xImgShape = image.shape[0]
        yImgShape = image.shape[0]

        # Shape of Output Convolution
        xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
        yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
        output = np.zeros_like(image)

        # Apply Equal Padding to All Sides
        if padding != 0:
            imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
            imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
            print(imagePadded)
        else:
            imagePadded = image

        # Iterate through image
        for y in range(image.shape[1]):
            # Exit Convolution
            if y > image.shape[1] - yKernShape:
                break
            # Only Convolve if y has gone down by the specified Strides
            if y % strides == 0:
                for x in range(image.shape[0]):
                    # Go to next row once kernel is out of bounds
                    if x > image.shape[0] - xKernShape:
                        break
                    try:
                        # Only Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            output[x, y] = abs((kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum())
                    except:
                        break

        return output

    def Q3_1(self):
        img = cv2.imread('./Q3_Image/Chihiro.jpg', 0)
        # Gaussian kernel
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        # Convolution
        gaussian_img = self.convolve2d(img, gaussian_kernel)
        
        cv2.namedWindow('Grayscale', cv2.WINDOW_NORMAL)
        cv2.imshow('Grayscale', img)
        cv2.namedWindow('Gaussian Blur', cv2.WINDOW_NORMAL)
        cv2.imshow('Gaussian Blur', gaussian_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Q3_2(self):
        img = cv2.imread('./Q3_Image/Chihiro.jpg', 0)
        # Gaussian kernel
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        # Sobel X kernel
        sobel_x_kernel = [[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]]
        # Convolution
        gaussian_img = self.convolve2d(img, gaussian_kernel)
        sobel_x_img = self.convolve2d(gaussian_img, sobel_x_kernel)
        sobel_x_img = cv2.normalize(sobel_x_img, None, 0, 255, cv2.NORM_MINMAX)

        # cv2.namedWindow('Gaussian Blur', cv2.WINDOW_NORMAL)
        # cv2.imshow('Gaussian Blur', gaussian_img)
        cv2.namedWindow('Sobel X', cv2.WINDOW_NORMAL)
        cv2.imshow('Sobel X', sobel_x_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Q3_3(self):
        img = cv2.imread('./Q3_Image/Chihiro.jpg', 0)
        # Gaussian kernel
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()        

        # Sobel Y kernel
        sobel_y_kernel = [[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]]
        # Convolution
        gaussian_img = self.convolve2d(img, gaussian_kernel)
        sobel_y_img = self.convolve2d(gaussian_img, sobel_y_kernel)
        sobel_y_img = cv2.normalize(sobel_y_img, None, 0, 255, cv2.NORM_MINMAX)

        # cv2.namedWindow('Gaussian Blur', cv2.WINDOW_NORMAL)
        # cv2.imshow('Gaussian Blur', gaussian_img)
        cv2.namedWindow('Sobel Y', cv2.WINDOW_NORMAL)
        cv2.imshow('Sobel Y', sobel_y_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Q3_4(self):
        img = cv2.imread('./Q3_Image/Chihiro.jpg', 0)
        # Gaussian kernel
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        # Sobel X kernel
        sobel_x_kernel = [[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]]
        # Sobel Y kernel
        sobel_y_kernel = [[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]]
        
        # Convolution
        gaussian_img = self.convolve2d(img, gaussian_kernel)
        sobel_x_img = self.convolve2d(gaussian_img, sobel_x_kernel)
        # sobel_x_img = cv2.normalize(sobel_x_img, None, 0, 255, cv2.NORM_MINMAX)
        sobel_y_img = self.convolve2d(gaussian_img, sobel_y_kernel)
        # sobel_y_img = cv2.normalize(sobel_y_img, None, 0, 255, cv2.NORM_MINMAX)

        # Magnitude
        magnitude_img = np.zeros(sobel_x_img.shape, dtype='uint8')
        for x in range(sobel_x_img.shape[0]):
            for y in range(sobel_x_img.shape[1]):
                magnitude_img[x, y] = (sobel_x_img[x, y]**2 + sobel_y_img[x, y]**2)**0.5
        magnitude_img = cv2.normalize(magnitude_img, None, 0, 255, cv2.NORM_MINMAX)

        # cv2.namedWindow('Sobel X', cv2.WINDOW_NORMAL)
        # cv2.imshow('Sobel X', sobel_x_img)
        # cv2.namedWindow('Sobel Y', cv2.WINDOW_NORMAL)
        # cv2.imshow('Sobel Y', sobel_y_img)
        cv2.namedWindow('Magnitude', cv2.WINDOW_NORMAL)
        cv2.imshow('Magnitude', magnitude_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def Q4(self, rotate, scale, tx, ty):
        # ref: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
        print('rotation: {}\tscaling: {}\ttx: {}\tty: {}'.format(
            rotate, scale, tx, ty))
        if not len(rotate) or not len(scale) or not len(tx) or not len(ty):
            return
        img = cv2.imread('./Q4_Image/Parrot.png')
        center = (160, 84)

        # Translation
        translation_M = np.float32([[1, 0, float(tx)], [0, 1, float(ty)]])
        transform_img = cv2.warpAffine(img, translation_M, (img.shape[1], img.shape[0]))
        # Rotation
        rotate_M = cv2.getRotationMatrix2D(center, float(rotate), float(scale))
        transform_img = cv2.warpAffine(transform_img, rotate_M, (img.shape[1], img.shape[0]))
        # Append two images
        img = np.hstack((img, transform_img))

        cv2.namedWindow('Transforms', cv2.WINDOW_NORMAL)
        cv2.imshow('Transforms', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = PyMainWindow()
    ui.setupUi(window)
    ui.show()
    sys.exit(app.exec_())
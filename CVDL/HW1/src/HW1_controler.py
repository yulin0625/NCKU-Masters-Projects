from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import os
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as img
import torch
from torchsummary import summary
from torchvision.transforms import transforms
from torchvision.models import vgg19_bn
import myVGG19

from HW1_UI import Ui_MainWindow  # Modify UI as your ui_file.py name
from myVGG19 import MyVGG19_BN

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # in python3, super(Class, self).xxx = super().xxx
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

        self.dataset_dir = ""
        self.images = []
        self.objPoints = dict()
        self.imgPoints = dict()
        self.intrinsicMatrix = dict()
        self.distortionMatrix = dict()
        self.rvecs = dict()
        self.tvecs = dict()
        self.extrinsicMatrix = dict()
        self.extrinsicMatrix["Q1"] = []
        self.extrinsicMatrix["Q2"] = []
        # Q3
        self.Q3Image_L = None
        self.Q3Image_R = None
        # Q4
        self.Q4Image1 = None
        self.Q4Image2 = None
        # Q5
        self.Q5_images_path = []
        self.model = MyVGG19_BN(num_classes=10, dropout=0.5)
        # self.model.load_state_dict(torch.load("./models/VGG19_bn_cifar10_state_dict.pth"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load("./models/VGG19_bn_cifar10_state_dict_20231104_135438_0.8711.pth", map_location=device))



    def setup_control(self):
        # TODO
        self.ui.btn_Load_Folder.clicked.connect(self.load_folder)
        self.ui.btn_Load_Image_L.clicked.connect(self.load_image_L)
        self.ui.btn_Load_Image_R.clicked.connect(self.load_image_R)
        # Q1
        self.ui.btn_Q1_1_Find_Corners.clicked.connect(self.btn_Q1_1_Find_Corners_clicked)

        self.ui.btn_Q1_2_Find_Intrinsic.clicked.connect(self.btn_Q1_2_Find_Intrinsic_clicked)

        self.ui.btn_Q1_3_Find_Extrinsic.clicked.connect(self.btn_Q1_3_Find_Extrinsic_clicked)
        self.ui.btn_Q1_4_Find_Distortion.clicked.connect(self.btn_Q1_4_Find_Distortion_clicked)
        self.ui.btn_Q1_5_Show_Result.clicked.connect(self.btn_Q1_5_Show_Result_clicked)

        # Q2
        self.ui.btn_Q2_1_Show_Word_on_Board.clicked.connect(self.btn_Q2_1_Show_Word_on_Board_clicked)
        self.ui.btn_Q2_2_Show_Word_on_Vertically.clicked.connect(self.btn_Q2_2_Show_Word_on_Vertically_clicked)

        # Q3
        self.ui.btn_Q3_1_Stereo_Disparity_Map.clicked.connect(self.btn_Q3_1_Stereo_Disparity_Map_clicked)

        # Q4
        self.ui.btn_Q4_Load_Image1.clicked.connect(self.btn_Q4_Load_Image1_clicked)
        self.ui.btn_Q4_Load_Image2.clicked.connect(self.btn_Q4_Load_Image2_clicked)
        self.ui.btn_Q4_1_Keypoints.clicked.connect(self.btn_Q4_1_Keypoints_clicked)
        self.ui.btn_Q4_2_Matched_Keypoints.clicked.connect(self.btn_Q4_2_Matched_Keypoints_clicked)

        # Q5
        self.ui.btn_Q5_Load_Image.clicked.connect(self.btn_Q5_Load_Image_clicked)
        self.ui.btn_Q5_1_Show_Augmented_Images.clicked.connect(self.btn_Q5_1_Show_Augmented_Images_clicked)
        self.ui.btn_Q5_2_Show_Model_Structure.clicked.connect(self.btn_Q5_2_Show_Model_Structure_clicked)
        self.ui.btn_Q5_3_Show_Acc_and_Loss.clicked.connect(self.btn_Q5_3_Show_Acc_and_Loss_clicked)
        self.ui.btn_Q5_4_Inference.clicked.connect(self.btn_Q5_4_Inference_clicked)

    def load_folder(self):
        dir = QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
        if dir == "":
            print("No directory selected.")
            return
        else:
            self.dataset_dir = Path(dir)
            print("Directory selected:", self.dataset_dir)
            images_path = sorted(
                list(self.dataset_dir.glob("*.bmp")), key=lambda x: int(x.stem)
            )
            for image_path in images_path:
                print("load ", image_path)
                self.images.append(cv2.imread(str(image_path)))

    def check_image_loaded(self):
        if self.images == []:
            QMessageBox.about(self, "check", "No image,Please load folder first!")
            return False
        return True

    def load_image_L(self):
        image_path = QFileDialog.getOpenFileName(self, "Select Image", os.getcwd())
        print(image_path)
        if image_path[0] == "":
            print("No image selected.")
            return
        else:
            self.Q3Image_L = cv2.imread(image_path[0])
            print("Image selected:", image_path[0])

    def load_image_R(self):
        image_path = QFileDialog.getOpenFileName(self, "Select Image", os.getcwd()) # (path, type)
        if image_path[0] == "":
            print("No image selected.")
            return
        else:
            self.Q3Image_R = cv2.imread(image_path[0])
            print("Image selected:", image_path[0])

    def btn_Q1_1_Find_Corners_clicked(self):
        if self.check_image_loaded() == False:
            return
        self.objPoints["Q1"], self.imgPoints["Q1"] = self.find_corners(
            self.images, True
        )

    def find_corners(self, images, show=False):
        winSize = (5, 5)  # 計算亞像素角點時搜尋window邊長的一半, 區域大小為 N x N, N = winSize x 2 + 1
        zeroZone = (-1, -1)  # 搜尋區域中間的dead region邊長的一半，通常忽略(-1, -1)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )  # termination criteria

        # number of  corners x,y
        x = 11
        y = 8

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((x * y, 3), np.float32)
        objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for image in images:
            img = image.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (x, y), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, winSize, zeroZone, criteria)
                if corners2 is not None:
                    imgpoints.append(corners2)
                else:
                    imgpoints.append(corners)
                    
                if show:
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (x, y), corners2, ret)
                    img = cv2.resize(img, (768, 768))
                    cv2.imshow("img", img)
                    cv2.waitKey(250)
                    cv2.destroyAllWindows()

        return objpoints, imgpoints

    def btn_Q1_2_Find_Intrinsic_clicked(self):
        if self.check_image_loaded() == False:
            return
        if self.objPoints.get("Q1") is None:
            self.objPoints["Q1"], self.imgPoints["Q1"] = self.find_corners(
                self.images, False
            )

        (
            ret,
            self.intrinsicMatrix["Q1"],
            self.distortionMatrix["Q1"],
            self.rvecs["Q1"],
            self.tvecs["Q1"],
        ) = self.calibrate_camera(
            self.images, self.objPoints["Q1"], self.imgPoints["Q1"]
        )
        print("\nIntrinsicMatrix:\n", self.intrinsicMatrix["Q1"])

    def calibrate_camera(self, Images, objPoints, imgPoints):
        ret, intrinsicMatrix, distortionMatrix, rvecs, tvecs = cv2.calibrateCamera(
            objPoints, imgPoints, (Images[0].shape[1], Images[0].shape[0]), None, None
        )

        return ret, intrinsicMatrix, distortionMatrix, rvecs, tvecs

    def btn_Q1_3_Find_Extrinsic_clicked(self):
        if self.check_image_loaded() == False:
            return
        choicePic = self.ui.comboBox_Q1_3.currentText()
        print("Chose image:", choicePic)

        if self.objPoints.get("Q1") is None:
            self.objPoints["Q1"], self.imgPoints["Q1"] = self.find_corners(
                self.images, False
            )
        if self.intrinsicMatrix.get("Q1") is None:
            (
                ret,
                self.intrinsicMatrix["Q1"],
                self.distortionMatrix["Q1"],
                self.rvecs["Q1"],
                self.tvecs["Q1"],
            ) = self.calibrate_camera(
                self.images, self.objPoints["Q1"], self.imgPoints["Q1"]
            )

        self.extrinsicMatrix["Q1"] = self.find_extrinsic(
            self.rvecs["Q1"], self.tvecs["Q1"]
        )
        print("\nExtrinsic\n", self.extrinsicMatrix["Q1"][int(choicePic) - 1])

    def find_extrinsic(self, rvecs, tvecs):
        extrinsicMatrix = []
        for i in range(len(rvecs)):
            rvec, temp = cv2.Rodrigues(rvecs[i])
            extrinsicMatrix.append(np.concatenate((rvec, tvecs[i]), axis=1))
        return extrinsicMatrix

    def btn_Q1_4_Find_Distortion_clicked(self):
        if self.check_image_loaded() == False:
            return
        if self.objPoints.get("Q1") is None:
            self.objPoints["Q1"], self.imgPoints["Q1"] = self.find_corners(
                self.images, False
            )
        if self.distortionMatrix.get("Q1") is None:
            (
                ret,
                self.intrinsicMatrix["Q1"],
                self.distortionMatrix["Q1"],
                self.rvecs["Q1"],
                self.tvecs["Q1"],
            ) = self.calibrate_camera(
                self.images, self.objPoints["Q1"], self.imgPoints["Q1"]
            )

        print("\nDistortion\n", self.distortionMatrix["Q1"])

    def btn_Q1_5_Show_Result_clicked(self):
        if self.check_image_loaded() == False:
            return
        if self.objPoints.get("Q1") is None:
            self.objPoints["Q1"], self.imgPoints["Q1"] = self.find_corners(
                self.images, False
            )
        if self.distortionMatrix.get("Q1") is None:
            (
                ret,
                self.intrinsicMatrix["Q1"],
                self.distortionMatrix["Q1"],
                self.rvecs["Q1"],
                self.tvecs["Q1"],
            ) = self.calibrate_camera(
                self.images, self.objPoints["Q1"], self.imgPoints["Q1"]
            )

        for item in self.images:
            image = item.copy()
            h, w = image.shape[:2]
            cameraMtx, roi = cv2.getOptimalNewCameraMatrix(
                self.intrinsicMatrix["Q1"],
                self.distortionMatrix["Q1"],
                (w, h),
                1,
                (w, h),
            )
            dst = cv2.undistort(
                image,
                self.intrinsicMatrix["Q1"],
                self.distortionMatrix["Q1"],
                None,
                cameraMtx,
            )
            x, y, w, h = roi
            dst = dst[y : y + h, x : x + w]
            image = cv2.resize(image, (768, 768))
            dst = cv2.resize(dst, (768, 768))
            result = np.hstack((image, dst))
            cv2.imshow("Image", result)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

    def btn_Q2_1_Show_Word_on_Board_clicked(self):
        if self.check_image_loaded() == False:
            return

        if self.objPoints.get("Q2") is None:
            self.objPoints["Q2"], self.imgPoints["Q2"] = self.find_corners(
                self.images, False
            )

        ret, self.intrinsicMatrix["Q2"], self.distortionMatrix["Q2"], self.rvecs["Q2"], self.tvecs["Q2"] = self.calibrate_camera(self.images, self.objPoints["Q2"], self.imgPoints["Q2"])

        inputString = self.ui.plainTextEdit_Q2.toPlainText()
        print("Input string:", inputString)
        fs = cv2.FileStorage(
            "./Dataset_CvDl_Hw1/Q2_Image/Q2_Lib/alphabet_lib_onboard.txt",
            cv2.FILE_STORAGE_READ,
        )

        # 使用 OpenCV 的 FileStorage 來做 YAML 檔案格式的寫入與讀取
        charList = []
        for char in inputString:
            charList.append(fs.getNode(char).mat())

        self.augmented_reality(charList)

    def augmented_reality(self, charList):
        position = [[7, 5, 0], [4, 5, 0], [1, 5, 0], [7, 2, 0], [4, 2, 0], [1, 2, 0]]
        rvecs = np.float32(self.rvecs["Q2"])
        tvecs = np.float32(self.tvecs["Q2"])

        for i in range(len(self.images)):
            image = self.images[i].copy()
            for j in range(len(charList)):
                line = np.float32(charList[j] + position[j]).reshape(-1, 3)
                imgpts, jac = cv2.projectPoints(
                    line,
                    rvecs[i],
                    tvecs[i],
                    self.intrinsicMatrix["Q2"],
                    self.distortionMatrix["Q2"],
                )

                imgpts = imgpts.astype(int)
                for k in range(0, len(imgpts), 2):
                    image = cv2.line(image, tuple(imgpts[k].ravel()), tuple(imgpts[k + 1].ravel()), (255, 255, 0), 10)

            image = cv2.resize(image, (768, 768))
            cv2.imshow(f"Image{i}", image)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

    def btn_Q2_2_Show_Word_on_Vertically_clicked(self):
        if self.check_image_loaded() == False:
            return

        if self.objPoints.get("Q2") is None:
            self.objPoints["Q2"], self.imgPoints["Q2"] = self.find_corners(
                self.images, False
            )

        ret, self.intrinsicMatrix["Q2"], self.distortionMatrix["Q2"], self.rvecs["Q2"], self.tvecs["Q2"] = self.calibrate_camera(self.images, self.objPoints["Q2"], self.imgPoints["Q2"])

        inputString = self.ui.plainTextEdit_Q2.toPlainText()
        fs = cv2.FileStorage(
            "./Dataset_CvDl_Hw1/Q2_Image/Q2_Lib/alphabet_lib_vertical.txt",
            cv2.FILE_STORAGE_READ,
        )

        # 使用 OpenCV 的 FileStorage 來做 YAML 檔案格式的寫入與讀取
        charList = []
        for char in inputString:
            charList.append(fs.getNode(char).mat())

        self.augmented_reality(charList)

    def btn_Q3_1_Stereo_Disparity_Map_clicked(self):
        if self.Q3Image_L is None :
            QMessageBox.about(self, "check", "No image,Please load image_L")
            return
        if self.Q3Image_R is None :
            QMessageBox.about(self, "check", "No image,Please load image_R")
            return
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        imgL_g = cv2.cvtColor(self.Q3Image_L, cv2.COLOR_BGR2GRAY) # Gray
        imgR_g = cv2.cvtColor(self.Q3Image_R, cv2.COLOR_BGR2GRAY) # Gray
        disparity = stereo.compute(imgL_g, imgR_g)
        disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        def onclick(event):
            dis = disparity[int(event.ydata)][int(event.xdata)]
            if dis ==0:
                return
            x = event.xdata  - dis
            y = event.ydata
            circle = plt.Circle((x, y), 10, color='red')
            for patch in ax2.patches:
                patch.remove()
            ax2.add_patch(circle)
            figR.canvas.draw()

        figL = plt.figure(figsize=(12, 8))
        imgL = cv2.cvtColor(self.Q3Image_L, cv2.COLOR_BGR2RGB) # RGB
        ax1 = figL.add_subplot(111)
        plt.imshow(imgL)
        figL.suptitle('ImageL', fontsize=16)

        figR = plt.figure(figsize=(12, 8))
        imgR = cv2.cvtColor(self.Q3Image_R, cv2.COLOR_BGR2RGB) # RGB
        ax2 = figR.add_subplot(111)
        plt.imshow(imgR)
        figL.suptitle('ImageR', fontsize=16)

        figG = plt.figure(figsize=(12, 8))
        ax3 = figG.add_subplot(111)
        plt.axis('off')
        plt.imshow(disparity, 'gray')
        
        plt.show()
        cid = figL.canvas.mpl_connect('button_press_event', onclick)

    # Q4
    def btn_Q4_Load_Image1_clicked(self):
        image_path = QFileDialog.getOpenFileName(self, "Select Image", os.getcwd())
        print(image_path)
        if image_path[0] == "":
            print("No image selected.")
            return
        else:
            self.Q4Image1 = cv2.imread(image_path[0], cv2.IMREAD_GRAYSCALE)
            print("Image selected:", image_path[0])

    def btn_Q4_Load_Image2_clicked(self):
        image_path = QFileDialog.getOpenFileName(self, "Select Image", os.getcwd())
        print(image_path)
        if image_path[0] == "":
            print("No image selected.")
            return
        else:
            self.Q4Image2 = cv2.imread(image_path[0], cv2.IMREAD_GRAYSCALE)
            print("Image selected:", image_path[0])

    def SIFT(self, grayImg):
        # grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Scale-Invariant Feature Transform,SIFT  
        sift = cv2.SIFT_create()

        # get the key point
        keypoints , descriptor = sift.detectAndCompute(grayImg, None)


        return keypoints, descriptor

    def btn_Q4_1_Keypoints_clicked(self):
        if self.Q4Image1 is None:
            QMessageBox.about(self, "check", "No image, Please load image1")
            return

        # image = cv2.cvtColor(self.Q4Image1, cv2.COLOR_BGR2GRAY)
        image = self.Q4Image1.copy()

        keypoints, descriptor  = self.SIFT(image)
        image_kp = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0))
        image_kp = cv2.resize(image_kp, (768, 768))
        cv2.imshow('Keypoints',image_kp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def btn_Q4_2_Matched_Keypoints_clicked(self):
        if self.Q4Image1 is None:
            QMessageBox.about(self, "check", "No image, Please load image1")
            return
        if self.Q4Image2 is None:
            QMessageBox.about(self, "check", "No image, Please load image2")
            return

        # img1 = cv2.cvtColor(self.Q4Image1, cv2.COLOR_BGR2GRAY)
        img1 = self.Q4Image1.copy()
        # img2 = cv2.cvtColor(self.Q4Image2, cv2.COLOR_BGR2GRAY)
        img2 = self.Q4Image2.copy()

        keypoints1 , descriptor1 = self.SIFT(img1)
        keypoints2 , descriptor2  = self.SIFT(img2)

        bf = cv2.BFMatcher()
        match = bf.knnMatch(descriptor1, descriptor2, k=2)
        goodMatch = []
        for m, n in match:
            if m.distance < 0.75 * n.distance:
                goodMatch.append(m)
        
        # orginalImg orginalkeypoints , Find Image, Find Image keypoints , Match array , 
        outImg = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, [goodMatch], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        outImg = cv2.resize(outImg, (768*2, 768))
        cv2.imshow('Keypoints',outImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def btn_Q5_Load_Image_clicked(self):
        self.Q5_images_path = [] # clear
        images_path = QFileDialog.getOpenFileNames(self, "Select demo Images", os.getcwd())
        print(images_path)
        if images_path[0] == "": # images_path = ([path], type)
            print("No image selected.")
            return
        else:
            print("Images selected:", images_path[0])
            for i in range(len(images_path[0])):
                self.Q5_images_path.append(Path(images_path[0][i]))
                print("Image selected:", images_path[0][i])


    # def btn_Q5_Load_Image_clicked(self):
    #     dir = QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
    #     if dir == "":
    #         print("No directory selected.")
    #         return
    #     else:
    #         self.Q5dataset_dir = Path(dir)
    #         print("Directory selected:", self.Q5dataset_dir)

    def data_augmentation(saelf, image):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
        ])
        return transform(image)

    def btn_Q5_1_Show_Augmented_Images_clicked(self):
        images = []
        labels = []
        for img_path in self.Q5_images_path:
            labels.append(img_path.stem)
            img = Image.open(img_path)
            img = self.data_augmentation(img)
            images.append(img)

        fig = plt.figure()
        for i in range(len(images)):
            img = images[i]
            label = labels[i]
            plt.subplot(3, 3, i + 1)
            plt.imshow(img)
            plt.title(label)
            plt.axis("off")

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
        plt.show()

    # def btn_Q5_1_Show_Augmented_Images_clicked(self):
    #     images_path = self.Q5dataset_dir.glob("*.png")
    #     images = []
    #     labels = []
    #     for img_path in images_path:
    #         labels.append(img_path.stem)
    #         img = Image.open(img_path)
    #         img = self.data_augmentation(img)
    #         images.append(img)

    #     fig = plt.figure()
    #     for i in range(len(images)):
    #         img = images[i]
    #         label = labels[i]
    #         plt.subplot(3, 3, i + 1)
    #         plt.imshow(img)
    #         plt.title(label)
    #         plt.axis("off")

    #     plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    #     plt.show()

    def btn_Q5_2_Show_Model_Structure_clicked(self):
        summary(self.model, (3, 32, 32))

    def btn_Q5_3_Show_Acc_and_Loss_clicked(self):
        fig = plt.figure(figsize=(20, 10))
        plt.subplot(121)                   # 建立 2x2 子圖表的左上方圖表
        # acc = img.imread("./log/vgg19_bn_Accuracy.jpg")
        acc = img.imread("./log/vgg19_bn_Accuracy_20231104_135438.jpg")

        plt.imshow(acc)
        plt.axis("off")

        plt.subplot(122) 
        loss = img.imread("./log/vgg19_bn_Loss_20231104_135438.jpg")
        plt.imshow(loss)
        plt.axis("off")

        plt.show()

    def btn_Q5_4_Inference_clicked(self):
        inference_img = Image.open(self.Q5_images_path[0])

        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        inference_img = transform(inference_img)
        pixmap = QtGui.QPixmap(str(self.Q5_images_path[0]))
        pixmap = pixmap.scaled(128, 128, QtCore.Qt.KeepAspectRatio)
        self.ui.label_Q5_inference_Image.setPixmap(pixmap)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_normalized = inference_img.unsqueeze_(0)
        img_normalized = img_normalized.to(device)


        with torch.no_grad():
            test_model = self.model.to(device)
            test_model.eval()
            output = test_model(img_normalized)
            probs = torch.softmax(output, dim=1)
            # _, pred = torch.max(probs, dim=1)
            pred = probs.data.cpu().numpy().argmax()
            pred_class_name = classes[pred]
            print(f"Predicted Class: {pred_class_name}")
            self.ui.label_Q5_predict.setText(f"predict = {pred_class_name}")
            x = [i for i in range(len(classes))]
            fig = plt.figure(figsize=(5, 5))
            plt.bar(x, probs.data.cpu().numpy()[0], tick_label=classes)
            plt.title(f"Probability of each class")
            plt.ylim(0, 1)
            plt.xticks(rotation = 45)
            plt.xlabel("Class")
            plt.ylabel("Probability")
            plt.show()
            print(probs.data.cpu().numpy()[0])
    

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QApplication, QMainWindow, QFileDialog, QMessageBox, QLabel, QHBoxLayout, QTabWidget, QComboBox
from CC_ui import Ui_MainWindow
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import PyQt5.QtCore as QtCore
import cv2
import numpy as np
import os
from imutils.perspective import four_point_transform
import CC_IQA
import subprocess
from ultralytics import YOLO
from PIL import Image
from VideoRecover import Video
from AnalyzeDisplay import ColorBoardCanvas, ColorBoardDeltaECanvas, tanaAnalyze
from Webcam import Webcam
import shutil

# 設置環境變數
from PyQt5.QtCore import QLibraryInfo
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

class StartPage(QWidget, QtCore.QObject):
    image_uploaded = QtCore.pyqtSignal(str)

    # 常數定義
    DEMO_WINDOW_SIZE = (801, 453)
    MAIN_WINDOW_SIZE = (1900, 1060)
    COLOR_BOARD_SIZE = (450, 450)
    TAB_SIZE = (1875, 500)
    
    def __init__(self):
        super().__init__()
        self.init_models()
        self.init_ui()
        self.setup_buttons()
        self.setup_selectors()
        self.setup_tabs()
        self.setup_layout()

    def init_models(self):
        self.times = 0
        self.yoloModel = YOLO("detection/best.pt")
        self.colorization_model = "neural-colorization/G.pth"
        self.firstTime_WaterNet = True
        self.firstTime_Colorization = True
        self.firstTime_Detection = True
        self.webcam_opened = False
        self.DEMO_SIZE = StartPage.DEMO_WINDOW_SIZE
        self.MAIN_SIZE = StartPage.MAIN_WINDOW_SIZE

    def init_ui(self):
        self.imageMainPage = QLabel(self)
        self.imageMainPage.setFixedSize(self.DEMO_SIZE[0], self.DEMO_SIZE[1])
        self.imageColorBoard = QLabel(self)
        self.imageColorBoard.setPixmap(QPixmap('Standard.png').scaled(StartPage.COLOR_BOARD_SIZE[0], StartPage.COLOR_BOARD_SIZE[1]))
        self.imageColorBlockAnalysis = QLabel(self)

    def setup_buttons(self):
        self.buttonWaterNet = QPushButton('waterNet', self)
        self.buttonColorization = QPushButton('colorization', self)
        self.buttonOpenImage = QPushButton('open image', self)
        self.buttonAnalyze = QPushButton('analyze', self)
        self.buttonDetection = QPushButton('detection', self)
        self.buttonOpenVideo = QPushButton('open video', self)
        self.buttonWebcam = QPushButton('webcam', self)
        self.connect_button_signals()

    def connect_button_signals(self):
        self.buttonWaterNet.clicked.connect(self.use_waterNet)
        self.buttonColorization.clicked.connect(self.use_colorization)
        self.buttonOpenImage.clicked.connect(self.open_image)
        self.buttonAnalyze.clicked.connect(self.open_Analyze)
        self.buttonDetection.clicked.connect(self.use_detection)
        self.buttonOpenVideo.clicked.connect(self.open_video)
        self.buttonWebcam.clicked.connect(self.use_webcam)

    def setup_selectors(self):
        self.colorization_selector = QComboBox(self)
        self.colorization_selector.addItems(["ocean", "people", "colorboard", "original"])
        self.colorization_selector.activated.connect(self.select_colorization)

        self.detection_selector = QComboBox(self)
        self.detection_selector.addItems(["fish", "colorBoard", "yolov8n", "yolov8x", "yolov8x-oiv7"])
        self.detection_selector.activated.connect(self.select_detection)

        self.webcam_selector = QComboBox(self)
        self.webcam_selector.addItems(["GRAY", "RGB"])
        self.webcam_selector.activated.connect(self.select_webcam_color)

    def setup_tabs(self):
        self.tab_widget = QTabWidget()
        self.tab_widget.setFixedSize(StartPage.TAB_SIZE[0], StartPage.TAB_SIZE[1])
        self.setup_tab3()
        self.setup_tab4()
        self.setup_tab5()

    def setup_tab3(self):
        tab3 = QWidget()
        layout_tab3 = QHBoxLayout()
        scroll_area_tab3 = QScrollArea()
        self.tab_image3 = tanaAnalyze(tab3, width=17, height=5, dpi=100, num_subplots=1)
        layout_tab3.addWidget(self.tab_image3)
        tab3.setLayout(layout_tab3)
        scroll_area_tab3.setWidget(tab3)
        scroll_area_tab3.setAlignment(Qt.AlignCenter)
        self.tab_widget.addTab(scroll_area_tab3, "人工抓取分析結果1.Delta E")

    def setup_tab4(self):
        tab4 = QWidget()
        layout_tab4 = QHBoxLayout()
        scroll_area_tab4 = QScrollArea()
        self.tab_image4 = ColorBoardCanvas(tab4, width=17, height=5, dpi=100, num_subplots=3)
        layout_tab4.addWidget(self.tab_image4)
        tab4.setLayout(layout_tab4)
        scroll_area_tab4.setWidget(tab4)
        scroll_area_tab4.setAlignment(Qt.AlignCenter)
        self.tab_widget.addTab(scroll_area_tab4, "人工抓取分析結果2.色塊")

    def setup_tab5(self):
        tab5 = QWidget()
        layout_tab5 = QHBoxLayout()
        scroll_area_tab5 = QScrollArea()
        self.tab_image5 = ColorBoardDeltaECanvas(tab5, width=17, height=5, dpi=100, num_subplots=3)
        layout_tab5.addWidget(self.tab_image5)
        tab5.setLayout(layout_tab5)
        scroll_area_tab5.setWidget(tab5)
        scroll_area_tab5.setAlignment(Qt.AlignCenter)
        self.tab_widget.addTab(scroll_area_tab5, "人工抓取分析結果3.長條圖")

    def setup_layout(self):
        layout = QHBoxLayout()
        layout_left = QVBoxLayout()
        layout_left.addWidget(self.imageMainPage)
        layout_left.addLayout(self.create_button_selector_layout())
        layout_right = QVBoxLayout()
        layout_right.addWidget(self.imageColorBoard)
        layout_right.addWidget(self.imageColorBlockAnalysis)
        layout_right.addStretch()
        layout.addLayout(layout_left)
        layout.addLayout(layout_right)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(self.tab_widget)

        self.setLayout(main_layout)
        self.setWindowTitle("Start Page")
        self.setFixedSize(*self.MAIN_SIZE)

    def create_button_selector_layout(self):
        buttons_and_selectors = [
            (self.buttonWaterNet, None),
            (self.buttonColorization, self.colorization_selector),
            (self.buttonOpenImage, None),
            (self.buttonAnalyze, None),
            (self.buttonOpenVideo, None),
            (self.buttonDetection, self.detection_selector),
            (self.buttonWebcam, self.webcam_selector),
        ]
        button_selector_layout = QHBoxLayout()
        for button, selector in buttons_and_selectors:
            button_layout = QVBoxLayout()
            if button:
                button_layout.addWidget(button)
            if selector:
                button_layout.addWidget(selector)
            button_layout.addStretch()
            button_selector_layout.addLayout(button_layout)
        return button_selector_layout
    
    def select_colorization(self):
        select = self.colorization_selector.currentText()
        if select == "ocean":
            self.colorization_model = "neural-colorization/G_water.pth"
        elif select == "people":
            self.colorization_model = "neural-colorization/G_people_v1.pth"
        elif select == "colorboard":
            self.colorization_model = "neural-colorization/G_colorboard.pth"
        elif select == "original":
            self.colorization_model = "neural-colorization/G.pth"
        self.firstTime_Colorization = True
    
    def select_detection(self):
        select = self.detection_selector.currentText()
        if select == "fish":
            self.yoloModel = YOLO("detection/betterdetection/best.pt")
        elif select == "colorBoard":
            self.yoloModel = YOLO("detection/best.pt")
        elif select == "yolov8n":
            self.yoloModel = YOLO("detection/yolov8n.pt")
        elif select == "yolov8x":
            self.yoloModel = YOLO("detection/yolov8x.pt")
        elif select == "yolov8x-oiv7":
            self.yoloModel = YOLO("detection/yolov8x-oiv7.pt")
        self.firstTime_Detection = True
    
    def select_webcam_color(self):
        select = self.webcam_selector.currentText()
        if select == "GRAY":
            self.webcam_color = "GRAY"
        elif select == "RGB":
            self.webcam_color = "RGB"
    
    def use_webcam(self): 
        # 輸入新影像，所以將還原次數重置
        self.firstTime_WaterNet = True
        self.firstTime_Colorization = True
        self.firstTime_Detection = True
        self.imageRestored_path = ''
        # 重置滑鼠追蹤事件
        self.imageMainPage.setMouseTracking(False)
        
        if not self.webcam_opened:
            # 創建一個Webcam對象，將self.imageMainPage傳入
            self.webcam = Webcam(self.imageMainPage, self.webcam_color)
            self.webcam.start_capture()
            self.webcam_opened = True
        else:
            # 如果Webcam已經開啟，可以在這裡執行關閉Webcam的操作
            self.webcam.stop_capture()
            self.webcam_opened = False
            self.webcam = Webcam(self.imageMainPage, self.webcam_color)
            self.webcam.start_capture()
            self.webcam_opened = True

    def capture(self):
        # 使用 Webcam 類的 save_current_frame 方法來保存畫面
        webcam_image_path = 'res/webcam_capture.jpg'
        self.webcam.save_current_frame(webcam_image_path)

        # 關閉 Webcam
        self.webcam.stop_capture()
        self.webcam_opened = False

        # 更新 img_path
        self.img_path = webcam_image_path
        self.imageOriginal = cv2.resize(cv2.imread(self.img_path), (self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
        self.imageMainPage.setPixmap(QPixmap(self.img_path).scaled(self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
        self.imageRestored = self.imageOriginal.copy()
        
        
    def use_waterNet(self):
        # 鏡頭處理
        if self.webcam_opened:
            self.capture()

        def call_inference(): # inference.py (WaterNet)
            # 設定參數
            inference_path = os.path.expanduser("waternet/inference.py")
            source_path = os.path.expanduser(self.img_path)
            weights_path = os.path.expanduser("waternet/weights/last.pt")
            output_path = os.path.expanduser('res/')

            #使用subprocess.call()來呼叫inference.py程式
            subprocess.call([
                "python3", inference_path,
                "--source", source_path,
                "--weights", weights_path,
                "--output", output_path,
            ])
        try:
            if self.firstTime_WaterNet == True and self.img_path != None:
                
                # lazy loaging
                # 並設置大小
                self.imageMainPage.setPixmap(QPixmap('res/loading.jpeg').scaled(self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
                QApplication.processEvents() # 強制更新畫面

                # 運行waterNet
                call_inference()
                self.firstTime_WaterNet = False
                # 取得self.img_path的檔名
                name = os.path.basename(self.img_path)
                
                # 將檔名改成 waterNet.jpg 以符合 imageRestored_path的預設位置
                shutil.copy('res/'+name, 'res/waterNet.jpg')

            self.imageRestored_path = 'res/waterNet.jpg'
            self.imageRestored = cv2.imread(self.imageRestored_path)
            # 顯示對比畫面
            self.image_show()
                
        except Exception as e:
            QMessageBox.information(self, "Error", "請先上傳圖片或是您的waterNet運行有錯誤", QMessageBox.Ok)
            print("Error: 請先上傳圖片或是您的waterNet運行有錯誤，錯誤訊息如下：")
            print(e)
            return

    def use_colorization(self):
        # 鏡頭處理
        if self.webcam_opened:
            self.capture()

        def call_colorization():
            # 設定參數
            colorization_path = os.path.expanduser("neural-colorization/colorize.py")
            
            source_path = os.path.expanduser(self.img_path)
            # weights_path = os.path.expanduser("neural-colorization/G.pth")
            weights_path = self.colorization_model
            output_path = os.path.expanduser("res/colorization.jpg")

            #使用subprocess.call()來呼叫colorization.py程式
            subprocess.call([
                "python3", colorization_path,
                "-i", source_path,
                "-m", weights_path,
                "-o", output_path,
                "--gpu", "-1",
            ])
        try:
            if self.firstTime_Colorization == True and self.img_path != None:
                # lazy loaging
                # 並設置大小
                self.imageMainPage.setPixmap(QPixmap('res/loading.jpeg').scaled(self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
                QApplication.processEvents() # 強制更新畫面

                # 運行colorization
                call_colorization()
                self.firstTime_Colorization = False
            self.imageRestored_path = 'res/colorization.jpg'
            self.imageRestored = cv2.imread(self.imageRestored_path)
            self.image_show()
        except Exception as e:
            QMessageBox.information(self, "Error", "請先上傳圖片或是您的colorization運行有錯誤", QMessageBox.Ok)
            print("Error: 請先上傳圖片或是您的colorization運行有錯誤，錯誤訊息如下：")
            print(e)
            return
    
    def use_detection(self):
        # 鏡頭處理
        if self.webcam_opened:
            self.capture()
            
        # 設定參數
        try:
            left_source_path:str = os.path.expanduser(self.img_path)
            if self.img_path != None:
                # lazy loaging
                # 並設置大小
                self.imageMainPage.setPixmap(QPixmap('res/loading.jpeg').scaled(self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
                QApplication.processEvents() # 強制更新畫面

                if self.imageRestored_path != '':
                    right_source_path:str = os.path.expanduser(self.imageRestored_path)
                    results = self.yoloModel([left_source_path, right_source_path])
                else:
                    results = self.yoloModel(left_source_path)

                img_left_array = results[0].plot()
                img_left = Image.fromarray(img_left_array)
                img_left = img_left.resize((self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
                self.imageOriginal = np.array(img_left)
                if self.imageRestored_path != '':
                    img_right_array = results[1].plot()
                    img_right = Image.fromarray(img_right_array)
                    img_right = img_right.resize((self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
                    self.imageRestored = np.array(img_right)
                self.firstTime_Detection = False
            self.image_show()

        except Exception as e:
            QMessageBox.information(self, "Error", "請先上傳圖片或是物件偵測運行有錯誤", QMessageBox.Ok)
            print("Error: 請先上傳圖片或是物件偵測運行有錯誤，錯誤訊息如下：")
            print(e)
            return

    def open_image(self):
        try:
            current_path = os.path.abspath(__file__)
            parent_path = os.path.dirname(
                os.path.dirname(os.path.dirname(current_path)))
            dir_path = os.path.join(parent_path, 'input')
            openfile_name = QFileDialog.getOpenFileName(
                self, 'select images', dir_path, 'Excel files(*.jpg , *.png)')
        except Exception as e:
            print("Error: 請確認您的路徑是否有誤，錯誤訊息如下：")
            print(e)
            return
        if openfile_name[0] != '':
            self.img_path = openfile_name[0]
            self.imageOriginal = cv2.resize(cv2.imread(self.img_path), (self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
            self.imageMainPage.setPixmap(QPixmap(self.img_path).scaled(self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
            self.imageRestored = self.imageOriginal.copy()
            # 輸入新圖片，所以將還原次數重置
            self.firstTime_WaterNet = True
            self.firstTime_Colorization = True
            self.firstTime_Detection = True
            self.imageRestored_path = ''
            # 重置滑鼠追蹤事件
            self.imageMainPage.setMouseTracking(False)

    def open_Analyze(self):
        try:
            self.analyze_page = Analyze(self)
            if self.imageRestored_path == '':
                self.image_uploaded.emit(self.img_path)
            else:
                self.image_uploaded.emit(self.imageRestored_path)
            self.analyze_page.returnAnalyze.connect(self.get_return_data)  # 連接信號和槽
            self.analyze_page.show()

        except Exception as e:
            QMessageBox.information(self, "Error", "請先上傳圖片或是有其他路徑問題", QMessageBox.Ok)
            print("Error: 請先上傳圖片或是有其他路徑問題，錯誤訊息如下：")
            print(e)

    def open_video(self):
        try:
            self.video_page = Video()
            self.video_page.show()
        except Exception as e:
            print("Error: 請先上傳影片或是有其他路徑問題，錯誤訊息如下：")
            print(e)
    
    def update_image(self):
        pixmap = QPixmap('res/delta_e.png')
        scaled_pixmap = pixmap.scaled(435, 435, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imageColorBlockAnalysis.setPixmap(scaled_pixmap)
        # 立即更新畫面
        QApplication.processEvents()

    def image_show(self):
        if self.imageRestored_path != '':
            self.imageRestored = cv2.resize(
                self.imageRestored, (self.imageOriginal.shape[1], self.imageOriginal.shape[0]))

            # 生成一條紅色的線
            height, width, channels = self.imageOriginal.shape
            line_thickness = 2
            line_length = int(width / 2)
            line_color = (0, 0, 255)  # BGR格式，此處為紅色
            line_x = int(width / 2)
            line_start = (line_x, 0)
            line_end = (line_x, height)

            # 生成一張空白的黑色圖片，大小與self.imageOriginal相同
            merged_image = np.zeros((height, width, channels), dtype=np.uint8)

            # 將self.imageOriginal與self.imageRestored分別放在空白圖片的左半邊與右半邊
            merged_image[:, :line_end[0], :] = self.imageOriginal[:, :line_end[0], :]
            merged_image[:, line_end[0]:, :] = self.imageRestored[:, line_end[0]:, :]

            # 設置滑鼠追蹤事件
            self.imageMainPage.setMouseTracking(True)
            self.imageMainPage.mouseMoveEvent = self.on_mouse_move  # 設置滑鼠移動事件的回傳函數

            cv2.line(merged_image, line_start, line_end,
                    line_color, line_thickness)
        else:
            merged_image = self.imageOriginal.copy()

        # 將圖片色彩空間從BGR轉換成RGB
        merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)

        # 將圖片轉換成QImage格式
        height, width, channels = merged_image.shape
        bytesPerLine = channels * width
        qImg = QImage(merged_image.data, width, height,
                      bytesPerLine, QImage.Format_RGB888)

        # 將QImage格式的圖片顯示出來
        self.imageMainPage.setPixmap(QPixmap.fromImage(qImg))

    def on_mouse_move(self, event):
        # 取得滑鼠位置
        mouse_pos = event.pos()
        line_x = mouse_pos.x()

        # 更新線的位置
        line_start = (line_x, 0)
        line_end = (line_x, self.imageMainPage.pixmap().height())  # 使用imageMainPage的高度

        # 更新 merged_image
        self.imageRestored = cv2.resize(
            self.imageRestored, (self.imageOriginal.shape[1], self.imageOriginal.shape[0]))

        merged_image = np.zeros_like(self.imageOriginal)
        merged_image[:, :line_end[0], :] = self.imageOriginal[:, :line_end[0], :]
        merged_image[:, line_end[0]:, :] = self.imageRestored[:, line_end[0]:, :]

        # 繪製紅色線條
        line_thickness = 2
        line_color = (0, 0, 255)  # BGR格式，此為红色
        cv2.line(merged_image, line_start, line_end,
                line_color, thickness=line_thickness)

        # 將色彩空間從BGR轉換成RGB
        merged_image_rgb = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)

        # 將影像轉換成QImage格式
        height, width, channels = merged_image_rgb.shape
        qImg = QImage(merged_image_rgb.data, width, height,
                      width * channels, QImage.Format_RGB888)

        # 將QImage格式的影像顯示在imageMainPage標籤上
        self.imageMainPage.setPixmap(QPixmap.fromImage(qImg))

    # 接收回傳的資料並更新影像
    @QtCore.pyqtSlot(dict)
    def get_return_data(self, data):
        print(data['points'])
        self.return_points = data['points']
        self.tab_image3.update_figure(data)
        self.tab_image4.update_figure(data)
        self.tab_image5.update_figure(data)
        self.update_image()


class ImageHandler(QtCore.QObject):
    image_updated = QtCore.pyqtSignal(QPixmap)

    def __init__(self, cc_image):
        super().__init__()
        self.cc_image = cc_image
        self.ori_cc_img = None
        self.resize_cc_img = None

    def load_image(self, image_path):
        self.ori_cc_img = cv2.imread(image_path)
        self.ori_cc_img = cv2.cvtColor(self.ori_cc_img, cv2.COLOR_BGR2RGB)
        self.resize_cc_img = cv2.resize(self.ori_cc_img, (640, 480))
        self.update_image()

    def update_image(self):
        height, width, channel = self.resize_cc_img.shape
        bytes_per_line = 3 * width
        qimage = QImage(self.resize_cc_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        qpixmap = QPixmap.fromImage(qimage)
        self.image_updated.emit(qpixmap)

    def get_points(self, get_p):
        return self.cc_image.return_points(self.ori_cc_img, get_p)

    def reset_image(self):
        self.cc_image.reselect()


class ImageTransformer:
    @staticmethod
    def show_image(image_label, image, rgb=True):
        if rgb:
            rgb_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image.copy()
        label_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
        image_label.setPixmap(QPixmap.fromImage(label_image))

    @staticmethod
    def rotate_image(image):
        img = cv2.transpose(image)
        img = cv2.flip(img, 0)
        return img


class AnalyzeDataProcessor:
    def __init__(self, scale=0.5):
        self.scale = scale

    def set_scale(self, scale_text):
        self.scale = float(scale_text)

    def process_analyze_data(self, rect_img):
        return CC_IQA.cc_task(rect_img, self.scale)


class Analyze(QMainWindow, Ui_MainWindow, QtCore.QObject):
    returnAnalyze = QtCore.pyqtSignal(dict)

    def __init__(self, start_page):
        super(Analyze, self).__init__(start_page)
        self.setupUi(self)
        self.start_page = start_page
        self.init_components()
        self.connect_signals()

    def init_components(self):
        self.image_handler = ImageHandler(self.cc_image)
        self.data_processor = AnalyzeDataProcessor()
        self.image_transformer = ImageTransformer()
        self.cc_points = []
        self.get_p = False
        self.rect_img = None

    def connect_signals(self):
        self.PB_4points.clicked.connect(self.get_cc_points)
        self.PB_reset.clicked.connect(self.reset)
        self.PB_rot.clicked.connect(self.rotate_rect_image)
        self.PB_ok.clicked.connect(self.set_scale)
        self.PB_ok_2.clicked.connect(self.return_analyze_and_points)
        self.start_page.image_uploaded.connect(self.handle_image_uploaded)
        self.image_handler.image_updated.connect(self.cc_image.setPixmap)

    @QtCore.pyqtSlot(str)
    def handle_image_uploaded(self, image_path):
        self.image_handler.load_image(image_path)
        self.img_path = image_path

    def get_cc_points(self):
        self.get_p = True
        self.cc_points = self.image_handler.get_points(self.get_p)
        if not self.cc_points:
            self.show_error('The number of selected points is insufficient')
            return
        self.rect_img = four_point_transform(self.image_handler.ori_cc_img.copy(), np.array(self.cc_points))
        self.rect_img = cv2.resize(self.rect_img, (self.area_image.width(), self.area_image.height()))
        self.image_transformer.show_image(self.area_image, self.rect_img, rgb=False)

    def reset(self):
        self.image_handler.reset_image()

    def rotate_rect_image(self):
        self.rect_img = self.image_transformer.rotate_image(self.rect_img)
        self.rect_img = cv2.resize(self.rect_img, (self.area_image.width(), self.area_image.height()))
        self.image_transformer.show_image(self.area_image, self.rect_img, rgb=False)

    def set_scale(self):
        self.data_processor.set_scale(self.scale_text.text())

    def return_analyze_and_points(self):
        self.show_message("通知", "函數執行中...", False)
        data = self.data_processor.process_analyze_data(self.rect_img)
        self.label_C.setText(f"mean C: {data['mean_C']:.4f}")
        self.label_E.setText(f"mean E: {data['mean_E']:.4f}")
        pts = self.image_handler.get_points(self.get_p)
        if not pts:
            self.show_error('The number of selected points is insufficient')
            return
        analyze_data = data
        pts = list(map(tuple, pts))
        analyze_data['points'] = pts
        self.save_points(pts)
        self.returnAnalyze.emit(analyze_data)
        self.show_message("通知", "函數完成", True)

    def show_message(self, title, message, is_close):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.NoButton if not is_close else QMessageBox.Ok)
        msg.show()
        QApplication.processEvents()
        if is_close:
            msg.accept()
            self.close()

    def show_error(self, message):
        QMessageBox.information(self, 'Error', message, QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)

    def save_points(self, points):
        parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_file = os.path.join(parent_path, "points.txt")
        with open(output_file, 'w') as f:
            f.write(', '.join(str(p) for p in points))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    start_page = StartPage()
    start_page.show()
    sys.exit(app.exec_())

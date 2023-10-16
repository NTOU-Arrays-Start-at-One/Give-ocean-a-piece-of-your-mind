import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QApplication, QMainWindow, QFileDialog, QMessageBox, QLabel, QHBoxLayout, QTabWidget, QComboBox, QFormLayout
from CC_ui import Ui_MainWindow
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtGui import QImage, QPixmap, QMovie
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt, pyqtSlot
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
import threading

# 設置環境變數
from PyQt5.QtCore import QLibraryInfo
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

class StartPage(QWidget, QtCore.QObject):
    image_uploaded = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        #demo
        self.times = 0

        # model
        self.yoloModel = YOLO("detection/best.pt")
        self.colorization_model = "neural-colorization/G.pth"
        
        # 設置物件
        # 主圖片
        self.image_e = QLabel(self)
        self.image_e.setFixedSize(801, 453)
        
        # self.imgShow1 原圖片
        # self.imgShow2 被還原的圖片
        # self.img_path 原圖片的路徑
        # self.imgShow2_path 被還原的圖片的路徑
        # self.videoShow_path 原影片的路徑
        # self.videoShow2_path 被還原的影片的路徑

        # 當原圖已經被還原，不需要再還原一次
        self.firstTime_WaterNet = True
        self.firstTime_Colorization = True
        self.firstTime_Detection = True
		
		# 狀態變數，用於跟蹤Webcam是否開啟
        self.webcam_opened = False
        
        # 示範區塊大小
        self.DEMO_SIZE = (801, 453)
        self.MAIN_SIZE = (1900, 1060)

        # 指令與切換按鈕a, b, c, d
        button_a = QPushButton('waterNet', self)
        button_b = QPushButton('colorization', self)
        button_c = QPushButton('open image', self)
        button_d = QPushButton('analyze', self)
        button_e = QPushButton('detection',self)
        button_g = QPushButton('open video', self)
        button_h = QPushButton('webcam', self)

        self.colorization_selector = QComboBox(self)
        self.detection_selector = QComboBox(self)
        self.webcam_selector = QComboBox(self)
        
        # 副圖片
        self.image_f = QLabel(self)
        self.image_f.setPixmap(QPixmap('Standard.png').scaled(450,450))
        self.image_g = QLabel(self)

        # 版面配置
        layout = QHBoxLayout()
        layout_left = QVBoxLayout()
        layout_left.addWidget(self.image_e)
        layout_right = QHBoxLayout()

        # ----------- 按鈕配置 -----------
        # 建立按鈕和下拉選擇框，並使其對齊
        buttons_and_selectors = [
            (button_a, None),  # 填充空白
            (button_b, self.colorization_selector),
            (button_c, None),  # 填充空白
            (button_d, None),  # 填充空白
            (button_g, None),  # 填充空白
            (button_e, self.detection_selector),
            (button_h, self.webcam_selector)   
        ]

        # 取得按鈕和下拉選擇框對的最大寬度
        max_width = max(button.sizeHint().width() if button is not None else 0 for button, _ in buttons_and_selectors)

        # 建立佈局以容納每個按鈕和下拉選擇框對，從左到右
        button_selector_layout = QHBoxLayout()
        for button, selector in buttons_and_selectors:
            button_layout = QVBoxLayout()
            if button is not None:
                button_layout.addWidget(button)
            if selector is not None:
                button_layout.addWidget(selector)
            button_layout.addStretch()  # 用空白填滿剩餘空間，使它們上下對齊
            button_layout.setContentsMargins(0, 0, max_width - button_layout.sizeHint().width(), 0)
            button_selector_layout.addLayout(button_layout)

        layout_left.addLayout(button_selector_layout)

        layout_right.addWidget(self.image_f)
        layout_right.addWidget(self.image_g)
        layout_right.addStretch()

        layout.addLayout(layout_left)
        layout.addLayout(layout_right)
        # ----------- 按鈕配置 -----------
        
        # 下拉選單
        self.colorization_selector.addItem("test1")
        self.colorization_selector.addItem("test2")
        self.detection_selector.addItem("fish")
        self.detection_selector.addItem("colorBoard")
        self.detection_selector.addItem("yolov8n")
        self.detection_selector.addItem("yolov8x")
        self.detection_selector.addItem("yolov8x-oiv7")
        
        self.webcam_selector.addItem("GRAY")
        self.webcam_selector.addItem("RGB")
        
        self.colorization_selector.activated.connect(self.select_colorization)
        self.detection_selector.activated.connect(self.select_detection)
        self.webcam_selector.activated.connect(self.select_webcam_color)
        self.select_detection()
        self.select_colorization()
        self.select_webcam_color()
        
        # 頁籤
        tab_widget = QTabWidget()
        tab_widget.setFixedSize(1875, 500)

        # 第三頁
        tab3 = QWidget()
        layout_tab3 = QHBoxLayout()
        scroll_area_tab3 = QScrollArea()
        self.tab_image3 = tanaAnalyze(tab3, width=17, height=5, dpi=100, num_subplots=1)
        layout_tab3.addWidget(self.tab_image3)
        tab3.setLayout(layout_tab3)
        scroll_area_tab3.setWidget(tab3)
        scroll_area_tab3.setAlignment(Qt.AlignCenter)
        tab_widget.addTab(scroll_area_tab3, "人工抓取分析結果1.Delta E")

        # 第四頁
        tab4 = QWidget()
        layout_tab4 = QHBoxLayout()
        scroll_area_tab4 = QScrollArea()
        self.tab_image4 = ColorBoardCanvas(tab4, width=17, height=5, dpi=100, num_subplots=3)
        layout_tab4.addWidget(self.tab_image4)
        tab4.setLayout(layout_tab4)
        scroll_area_tab4.setWidget(tab4)
        scroll_area_tab4.setAlignment(Qt.AlignCenter)
        tab_widget.addTab(scroll_area_tab4, "人工抓取分析結果2.色塊")

        # 第五頁
        tab5 = QWidget()
        layout_tab5 = QHBoxLayout()
        scroll_area_tab5 = QScrollArea()
        self.tab_image5 = ColorBoardDeltaECanvas(tab5, width=17, height=5, dpi=100, num_subplots=3)
        layout_tab5.addWidget(self.tab_image5)
        tab5.setLayout(layout_tab5)
        scroll_area_tab5.setWidget(tab5)
        scroll_area_tab5.setAlignment(Qt.AlignCenter)
        tab_widget.addTab(scroll_area_tab5, "人工抓取分析結果3.長條圖")

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(tab_widget)

        self.setLayout(main_layout)
        self.setWindowTitle("Start Page")
        self.setFixedSize(self.MAIN_SIZE[0], self.MAIN_SIZE[1])

        # 點擊事件
        button_a.clicked.connect(self.use_waterNet)
        button_b.clicked.connect(self.use_colorization)
        button_c.clicked.connect(self.open_image)
        button_d.clicked.connect(self.open_Analyze)
        button_e.clicked.connect(self.use_detection)
        button_g.clicked.connect(self.open_video)
        button_h.clicked.connect(self.use_webcam)
    
    def select_colorization(self):
        select = self.colorization_selector.currentText()
        if select == "water":
            self.colorization_model = "neural-colorization/G.pth"
        elif select == "test2":
            self.colorization_model = "" # TODO
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
        self.imgShow2_path = ''
        # 重置滑鼠追蹤事件
        self.image_e.setMouseTracking(False)
        
        if not self.webcam_opened:
            # 創建一個Webcam對象，將self.image_e傳入
            self.webcam = Webcam(self.image_e, self.webcam_color)
            self.webcam.start_capture()
            self.webcam_opened = True
        else:
            # 如果Webcam已經開啟，可以在這裡執行關閉Webcam的操作
            self.webcam.stop_capture()
            self.webcam_opened = False
            self.webcam = Webcam(self.image_e, self.webcam_color)
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
        self.imgShow1 = cv2.resize(cv2.imread(self.img_path), (self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
        self.image_e.setPixmap(QPixmap(self.img_path).scaled(self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
        self.imgShow2 = self.imgShow1.copy()
        
        
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
                self.image_e.setPixmap(QPixmap('res/loading.jpeg').scaled(self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
                QApplication.processEvents() # 強制更新畫面

                # 運行waterNet
                call_inference()
                self.firstTime_WaterNet = False
                # 取得self.img_path的檔名
                name = os.path.basename(self.img_path)
                
                # 將檔名改成 waterNet.jpg 以符合 imgShow2_path的預設位置
                shutil.copy('res/'+name, 'res/waterNet.jpg')

            self.imgShow2_path = 'res/waterNet.jpg'
            self.imgShow2 = cv2.imread(self.imgShow2_path)
            # 顯示對比畫面
            self.image_show()
                
        except Exception as e:
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
                self.image_e.setPixmap(QPixmap('res/loading.jpeg').scaled(self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
                QApplication.processEvents() # 強制更新畫面

                # 運行colorization
                call_colorization()
                self.firstTime_Colorization = False
            self.imgShow2_path = 'res/colorization.jpg'
            self.imgShow2 = cv2.imread(self.imgShow2_path)
            self.image_show()
        except Exception as e:
            print("Error: 請先上傳圖片或是您的colorization運行有錯誤，錯誤訊息如下：")
            print(e)
            return
    
    def use_detection(self):
        # 鏡頭處理
        if self.webcam_opened:
            self.capture()
            
        # 設定參數
        left_source_path:str = os.path.expanduser(self.img_path)
        try:
            if self.img_path != None:
                # lazy loaging
                # 並設置大小
                self.image_e.setPixmap(QPixmap('res/loading.jpeg').scaled(self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
                QApplication.processEvents() # 強制更新畫面

                if self.imgShow2_path != '':
                    right_source_path:str = os.path.expanduser(self.imgShow2_path)
                    results = self.yoloModel([left_source_path, right_source_path])
                else:
                    results = self.yoloModel(left_source_path)

                img_left_array = results[0].plot()
                img_left = Image.fromarray(img_left_array)
                img_left = img_left.resize((self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
                self.imgShow1 = np.array(img_left)
                if self.imgShow2_path != '':
                    img_right_array = results[1].plot()
                    img_right = Image.fromarray(img_right_array)
                    img_right = img_right.resize((self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
                    self.imgShow2 = np.array(img_right)
                self.firstTime_Detection = False
            self.image_show()

        except Exception as e:
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
            self.imgShow1 = cv2.resize(cv2.imread(self.img_path), (self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
            self.image_e.setPixmap(QPixmap(self.img_path).scaled(self.DEMO_SIZE[0], self.DEMO_SIZE[1]))
            self.imgShow2 = self.imgShow1.copy()
            # 輸入新圖片，所以將還原次數重置
            self.firstTime_WaterNet = True
            self.firstTime_Colorization = True
            self.firstTime_Detection = True
            self.imgShow2_path = ''
            # 重置滑鼠追蹤事件
            self.image_e.setMouseTracking(False)

    def open_Analyze(self):
        try:
            self.analyze_page = Analyze(self)
            if self.imgShow2_path == '':
                self.image_uploaded.emit(self.img_path)
            else:
                self.image_uploaded.emit(self.imgShow2_path)
            self.analyze_page.returnAnalyze.connect(self.get_return_data)  # 連接信號和槽
            self.analyze_page.show()

        except Exception as e:
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
        self.image_g.setPixmap(scaled_pixmap)
        # 立即更新畫面
        QApplication.processEvents()

    def image_show(self):
        if self.imgShow2_path != '':
            self.imgShow2 = cv2.resize(
                self.imgShow2, (self.imgShow1.shape[1], self.imgShow1.shape[0]))

            # 生成一條紅色的線
            height, width, channels = self.imgShow1.shape
            line_thickness = 2
            line_length = int(width / 2)
            line_color = (0, 0, 255)  # BGR格式，此處為紅色
            line_x = int(width / 2)
            line_start = (line_x, 0)
            line_end = (line_x, height)

            # 生成一張空白的黑色圖片，大小與self.imgShow1相同
            merged_image = np.zeros((height, width, channels), dtype=np.uint8)

            # 將self.imgShow1與self.imgShow2分別放在空白圖片的左半邊與右半邊
            merged_image[:, :line_end[0], :] = self.imgShow1[:, :line_end[0], :]
            merged_image[:, line_end[0]:, :] = self.imgShow2[:, line_end[0]:, :]

            # 設置滑鼠追蹤事件
            self.image_e.setMouseTracking(True)
            self.image_e.mouseMoveEvent = self.on_mouse_move  # 設置滑鼠移動事件的回傳函數

            cv2.line(merged_image, line_start, line_end,
                    line_color, line_thickness)
        else:
            merged_image = self.imgShow1.copy()

        # 將圖片色彩空間從BGR轉換成RGB
        merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)

        # 將圖片轉換成QImage格式
        height, width, channels = merged_image.shape
        bytesPerLine = channels * width
        qImg = QImage(merged_image.data, width, height,
                      bytesPerLine, QImage.Format_RGB888)

        # 將QImage格式的圖片顯示出來
        self.image_e.setPixmap(QPixmap.fromImage(qImg))

    def on_mouse_move(self, event):
        # 取得滑鼠位置
        mouse_pos = event.pos()
        line_x = mouse_pos.x()

        # 更新線的位置
        line_start = (line_x, 0)
        line_end = (line_x, self.image_e.pixmap().height())  # 使用image_e的高度

        # 更新 merged_image
        self.imgShow2 = cv2.resize(
            self.imgShow2, (self.imgShow1.shape[1], self.imgShow1.shape[0]))

        merged_image = np.zeros_like(self.imgShow1)
        merged_image[:, :line_end[0], :] = self.imgShow1[:, :line_end[0], :]
        merged_image[:, line_end[0]:, :] = self.imgShow2[:, line_end[0]:, :]

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

        # 將QImage格式的影像顯示在image_e標籤上
        self.image_e.setPixmap(QPixmap.fromImage(qImg))

    # 接收回傳的資料並更新影像
    @QtCore.pyqtSlot(dict)
    def get_return_data(self, data):
        print(data['points'])
        self.return_points = data['points']
        self.tab_image3.update_figure(data)
        self.tab_image4.update_figure(data)
        self.tab_image5.update_figure(data)
        self.update_image()


class Analyze(QMainWindow, Ui_MainWindow, QtCore.QObject):
    returnAnalyze = QtCore.pyqtSignal(dict)

    def __init__(self, start_page):
        super(Analyze, self).__init__(start_page)
        self.setupUi(self)
        self.PB_4points.clicked.connect(self.get_cc_points)
        self.PB_reset.clicked.connect(self.reset)
        self.PB_rot.clicked.connect(self.rot_rect)
        self.PB_ok.clicked.connect(self.get_scale)
        self.PB_ok_2.clicked.connect(self.return_analyze_and_points)
        self.start_page = start_page
        self.start_page.image_uploaded.connect(self.handle_image_uploaded)
        self.img_path = ''
        self.cc_points = []
        self.get_p = False
        self.scale = 0.5

    @QtCore.pyqtSlot(str)
    def handle_image_uploaded(self, image_path):
        self.cc_image.reselect()
        self.img_path = image_path
        self.ori_cc_img = cv2.imread(self.img_path)
        self.ori_cc_img = cv2.cvtColor(self.ori_cc_img, cv2.COLOR_BGR2RGB)
        self.resize_cc_img = cv2.resize(self.ori_cc_img, (640, 480))
        # 將 OpenCV 的圖片轉換為 QImage
        height, width, channel = self.resize_cc_img.shape
        bytes_per_line = 3 * width
        qimage = QImage(self.resize_cc_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # 將 QImage 轉換為 QPixmap
        qpixmap = QPixmap.fromImage(qimage)
        # 在介面上顯示圖片
        self.cc_image.setPixmap(qpixmap)
    
    def get_cc_points(self):
        self.get_p = True
        self.cc_points = self.cc_image.return_points(self.ori_cc_img, self.get_p)
        if self.cc_points == False:
            QMessageBox.information(self, 'error', 'The number of selected points is insufficient',
                                    QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            return
        rect = four_point_transform(self.ori_cc_img.copy(), np.array(self.cc_points))
        self.rect_img = rect.copy()
        self.rect_img = cv2.resize(self.rect_img, (self.area_image.width(), self.area_image.height()))
        self.show_image(self.area_image, self.rect_img, rgb=False)

    def reset(self):
        self.cc_image.reselect()

    def show_image(self, image_label, image, rgb=True):
        if rgb is True:
            rgb_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image.copy()
        label_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
        image_label.setPixmap(QPixmap.fromImage(label_image))

    def rot_rect(self):
        img = cv2.transpose(self.rect_img)
        img = cv2.flip(img, 0)
        self.rect_img = img.copy()
        self.rect_img = cv2.resize(self.rect_img, (self.area_image.width(), self.area_image.height()))
        self.show_image(self.area_image, self.rect_img, rgb=False)

    def get_scale(self):
        tmp = self.scale_text.text()
        self.scale = float(tmp)

    def return_analyze_and_points(self):
        data = CC_IQA.cc_task(self.rect_img, self.scale)
        self.label_C.setText("mean C: {:.4f}".format(data["mean_C"]))
        self.label_E.setText("mean E: {:.4f}".format(data["mean_E"]))
        pts = self.cc_image.return_points(self.ori_cc_img, self.get_p)
        if pts == False:
            QMessageBox.information(self, 'error', 'The number of selected points is insufficient',
                                    QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            return
        analyze_data = data
        pts = list(map(tuple, pts))
        analyze_data['points'] = pts
        print(analyze_data)
        parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_file = os.path.join(parent_path, "points.txt")
        with open(output_file, 'w') as f:
            f.write(', '.join(str(p) for p in pts))
        self.returnAnalyze.emit(analyze_data)
        self.close()

    def on_exit_clicked(self):
        QApplication.exit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    start_page = StartPage()
    start_page.show()
    sys.exit(app.exec_())

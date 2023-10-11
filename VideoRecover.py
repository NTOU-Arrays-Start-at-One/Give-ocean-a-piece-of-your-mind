import os
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
import numpy as np
import subprocess


class Video(QWidget, QtCore.QObject):
    
    def __init__(self):
        super().__init__()

        # Create a QLabel to display the video frames
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)

        # Create buttons for opening and playing the video
        self.open_button = QPushButton('Open Video (waterNet)', self)
        self.test_button = QPushButton('Test Video', self)

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.open_button)
        layout.addWidget(self.test_button)

        # Connect button clicks to corresponding functions
        self.open_button.clicked.connect(self.open_video)
        self.test_button.clicked.connect(self.test_video)

        self.setLayout(layout)
        self.setWindowTitle("Video Player")

        # Video variables
        self.video_path = ""
        self.video_capture = None
        self.playing = False

    def open_video(self):
        try:
            current_path = os.path.abspath(__file__)
            parent_path = os.path.dirname(
                os.path.dirname(os.path.dirname(current_path)))
            dir_path = os.path.join(parent_path, 'input')
            video_file, _ = QFileDialog.getOpenFileName(
                self, 'Select Video File', dir_path, 'Video files (*.mp4 *.avi)')
            if video_file:
                self.video_path = video_file
                self.videoShow_path = self.video_path
                self.video_capture = cv2.VideoCapture(video_file)
                self.playing = False
            
        except Exception as e:
            print("Error: Unable to open the video file. Error message:")
            print(e)
        
        self.use_waterNet()
        self.video_show()
    
    def video_show(self, cap1, cap2):
        # 讀取兩段影片
        self.cap1 = cv2.VideoCapture(cap1)
        self.cap2 = cv2.VideoCapture(cap2)

        # 檢查影片大小是否一致，若不同可使用cv2.resize()函數進行調整
        self.width = 640
        self.height = 360
        self.fps = self.cap1.get(cv2.CAP_PROP_FPS)

        # 生成一條紅色的線
        self.line_thickness = 2
        self.line_length = int(self.width / 2)
        self.line_color = (0, 0, 255)  # BGR格式，此處為紅色
        self.line_x = int(self.width / 2)
        self.line_start = (self.line_x, 0)
        self.line_end = (self.line_x, self.height)

        # 設定疊加影片的起始時間
        self.start_time = 0
        self.cap1.set(cv2.CAP_PROP_POS_MSEC, self.start_time)
        self.cap2.set(cv2.CAP_PROP_POS_MSEC, self.start_time)

        # 創建一張空白的黑色圖片，大小與影片相同
        self.merged_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 設定更新影片畫面的計時器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / self.fps))  # 根據幀率設定更新間隔

        # 設置滑鼠事件的回調函數
        self.video_label.setMouseTracking(True)
        self.video_label.mouseMoveEvent = self.on_mouse_move
        
    def test_video(self):
        self.video_show('res/test_2.mp4', 'res/test_1.mp4')

    def update_frame(self):
        # 讀取兩個影格
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()

        # 若其中一個影片已讀取完畢，則退出迴圈
        if not ret1 or not ret2:
            self.cap1.release()
            self.cap2.release()
            self.timer.stop()
            return

        # 調整影格大小
        frame1 = cv2.resize(frame1, (self.width, self.height))
        frame2 = cv2.resize(frame2, (self.width, self.height))

        # 更新疊加後的圖片
        self.update_merged_image(frame1, frame2)

    def update_merged_image(self, frame1, frame2):
        # 將frame1與frame2分別放在空白圖片的左半邊與右半邊
        line_end = int(self.line_x)
        self.merged_image[:, :line_end, :] = frame1[:, :line_end, :]
        self.merged_image[:, line_end:, :] = frame2[:, line_end:, :]

        # 畫線
        self.line_start = (self.line_x, 0)
        self.line_end = (self.line_x, self.height)
        
        cv2.line(self.merged_image, self.line_start, self.line_end, self.line_color, thickness=self.line_thickness)

        # 將OpenCV圖像轉換為QImage並更新到QLabel
        merged_image_rgb = cv2.cvtColor(self.merged_image, cv2.COLOR_BGR2RGB)
        q_image = QImage(merged_image_rgb.data, self.width, self.height, self.width * 3, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def on_mouse_move(self, event):
        # 取得滑鼠位置
        mouse_pos = event.pos()
        self.line_x = mouse_pos.x()

    def use_waterNet(self):
        def call_inference(): # inference.py (WaterNet)
            inference_path = os.path.expanduser("waternet/inference.py")
            source_path = os.path.expanduser(self.video_path)
            weights_path = os.path.expanduser("waternet/weights/last.pt")
            output_path = os.path.expanduser('res/')

            subprocess.call([
                "python3", inference_path,
                "--source", source_path,
                "--weights", weights_path,
                "--output", output_path,
            ])
        try:
            if self.video_path != None:
                # lazy loaging
                # 並設置大小
                self.video_label.setPixmap(QPixmap('res/loading.jpeg').scaled(1024, 576))
                QApplication.processEvents() # 強制更新畫面
                # 運行waterNet
                call_inference()
                name = os.path.basename(self.video_path)
                os.rename('res/'+name, 'res/waterNet.mp4')
            self.videoShow2_path = 'res/waterNet.mp4'
            self.videoShow2 = cv2.imread(self.videoShow2_path)
            # 顯示對比畫面
            self.video_show(self.videoShow_path, self.videoShow2_path)

        except Exception as e:
            print("Error: 請先上傳圖片或是您的waterNet運行有錯誤，錯誤訊息如下：")
            print(e)
            return
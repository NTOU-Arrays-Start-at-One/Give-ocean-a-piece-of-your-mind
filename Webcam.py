import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class Webcam:
    def __init__(self, image_e):
        self.capture = cv2.VideoCapture(0)  # 打開第一個可用的攝像頭
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.image_e = image_e  # 將image_e傳入Webcam，以便更新圖像

    def start_capture(self):
        self.timer.start(30)  # 更新攝像頻率，這裡設置為每30毫秒更新一次畫面

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            # 將捕獲的圖像轉換為灰度圖像
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
            h, w = frame.shape  # 獲取灰度圖像的高度和寬度
            bytes_per_line = w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    
            # 顯示攝像頭畫面
            self.display_frame(qt_image)

    def save_current_frame(self, file_path):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 以獨特的名稱儲存當前鏡頭畫面
            cv2.imwrite(file_path, frame)

    def display_frame(self, image):
        # 將Qt影像設置到QLabel中
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(801, 453)
        self.image_e.setPixmap(pixmap)

    def stop_capture(self):
        self.timer.stop()
        self.capture.release()


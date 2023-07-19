import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QApplication, QMainWindow, QFileDialog, QMessageBox, QLabel, QHBoxLayout, QTabWidget
from CC_ui import Ui_MainWindow
from PyQt5.QtGui import QImage, QPixmap, QMovie
import PyQt5.QtCore as QtCore
import cv2
import numpy as np
import os
from imutils.perspective import four_point_transform
import CC_IQA
import subprocess

class StartPage(QWidget, QtCore.QObject):
    image_uploaded = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # 設置物件
        # 主圖片
        self.image_e = QLabel(self)
        
        # self.imgShow1 原圖片
        # self.imgShow2 被還原的圖片
        # self.img_path 原圖片的路徑
        # self.imgShow2_path 被還原的圖片的路徑

        # 當原圖已經被還原，不需要再還原一次
        self.firstTime_WaterNet = True
        self.firstTime_Colorization = True

        # 指令與切換按鈕a, b, c, d
        button_a = QPushButton('waterNet', self)
        button_b = QPushButton('colorization', self)
        button_c = QPushButton('open image', self)
        button_d = QPushButton('analyze', self)

        # 副圖片
        self.image_f = QLabel(self)
        self.image_f.setPixmap(QPixmap('Standard.png').scaled(600, 600))

        # 版面配置
        layout = QHBoxLayout()
        layout_left = QVBoxLayout()
        layout_right = QVBoxLayout()

        layout_left.addWidget(self.image_e)

        layout_button = QHBoxLayout()
        layout_button.addWidget(button_a)
        layout_button.addWidget(button_b)
        layout_button.addWidget(button_c)
        layout_button.addWidget(button_d)
        layout_left.addLayout(layout_button)

        layout_right.addStretch()
        layout_right.addWidget(self.image_f)

        layout.addLayout(layout_left)
        layout.addLayout(layout_right)

        # 頁籤
        tab_widget = QTabWidget()

        # 第一頁
        tab1 = QWidget()
        layout_tab1 = QHBoxLayout()
        self.tab_image1 = QLabel(tab1)
        self.tab_image1.setPixmap(QPixmap('res/Figure_1.png'))
        layout_tab1.addWidget(self.tab_image1)
        tab1.setLayout(layout_tab1)
        tab_widget.addTab(tab1, "Tab 1")

        # 第二頁
        tab2 = QWidget()
        layout_tab2 = QHBoxLayout()
        self.tab_image2 = QLabel(tab2)
        self.tab_image2.setPixmap(QPixmap('res/Figure_2.png'))
        layout_tab2.addWidget(self.tab_image2)
        tab2.setLayout(layout_tab2)
        tab_widget.addTab(tab2, "Tab 2")

        # 第三頁
        tab3 = QWidget()
        layout_tab3 = QHBoxLayout()
        self.tab_image3 = QLabel(tab3)
        self.tab_image3.setPixmap(QPixmap('res/Figure_3.png'))
        layout_tab3.addWidget(self.tab_image3)
        tab3.setLayout(layout_tab3)
        tab_widget.addTab(tab3, "Tab 3")

        # 第四頁
        tab4 = QWidget()
        layout_tab4 = QHBoxLayout()
        self.tab_image4 = QLabel(tab4)
        self.tab_image4.setPixmap(QPixmap('res/colorblock.png').scaled(864, 576))
        layout_tab4.addWidget(self.tab_image4)
        tab4.setLayout(layout_tab4)
        tab_widget.addTab(tab4, "人工抓取分析結果1.色塊")

        # 第五頁
        tab5 = QWidget()
        layout_tab5 = QHBoxLayout()
        self.tab_image5 = QLabel(tab5)
        self.tab_image5.setPixmap(QPixmap('res/k-means.png').scaled(1094, 576))
        layout_tab5.addWidget(self.tab_image5)
        tab4.setLayout(layout_tab5)
        tab_widget.addTab(tab5, "人工抓取分析結果2.長條圖")

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(tab_widget)

        self.setLayout(main_layout)
        self.setWindowTitle("Start Page")

        # 點擊事件
        button_a.clicked.connect(self.use_waterNet)
        button_b.clicked.connect(self.use_colorization)
        button_c.clicked.connect(self.open_image)
        button_d.clicked.connect(self.open_Analyze)

    def use_waterNet(self):

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
                self.image_e.setPixmap(QPixmap('res/loading.jpeg').scaled(1024, 576))
                QApplication.processEvents() # 強制更新畫面

                # 運行waterNet
                call_inference()
                self.firstTime_WaterNet = False
                # 取得self.img_path的檔名
                name = os.path.basename(self.img_path)
                # 將檔名改成 waterNet.jpg 以符合 imgShow2_path的預設位置
                os.rename('res/'+name, 'res/waterNet.jpg')

            self.imgShow2_path = 'res/waterNet.jpg'
            self.imgShow2 = cv2.imread(self.imgShow2_path)
            # 顯示對比畫面
            self.image_show()

        except Exception as e:
            print("Error: 請先上傳圖片或是您的waterNet運行有錯誤，錯誤訊息如下：")
            print(e)
            return

    def use_colorization(self):
        # Is first time?
        # Lazy Load
        # call

        self.imgShow2 = cv2.imread('res/colorization.jpg')
        self.image_show()

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
            self.imgShow1 = cv2.resize(cv2.imread(self.img_path), (1024, 576))
            self.image_e.setPixmap(QPixmap(self.img_path))
            self.imgShow2 = self.imgShow1.copy()
            # 輸入新圖片，所以將還原次數重置
            self.firstTime_WaterNet = True
            self.firstTime_Colorization = True
            self.imgShow2_path = ''

    def open_Analyze(self):
        try:
            self.analyze_page = Analyze(self)
            if self.imgShow2_path == '':
                self.image_uploaded.emit(self.img_path)
            else:
                self.image_uploaded.emit(self.imgShow2_path)
            self.analyze_page.returnPoints.connect(self.get_return_points)  # 連接信號和槽
            self.analyze_page.show()

        except Exception as e:
            print("Error: 請先上傳圖片或是有其他路徑問題，錯誤訊息如下：")
            print(e)

    def update_image(self):
        # 在圖片內容更改後，獲取新的圖片
        new_image = cv2.imread('res/colorblock.png')
        new_image1 = cv2.imread('res/k-means.png')
        # 將 OpenCV 的圖片轉換為 QImage
        height, width, _ = new_image.shape
        height1, width1, _ = new_image1.shape
        bytes_per_line = 3 * width
        bytes_per_line1 = 3 * width1
        qimage = QImage(new_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
        qimage1 = QImage(new_image1.data, width1, height1, bytes_per_line1, QImage.Format_BGR888)
        # 將 QImage 轉換為 QPixmap
        new_pixmap = QPixmap.fromImage(qimage)
        new_pixmap1 = QPixmap.fromImage(qimage1)
        self.tab_image4.setPixmap(new_pixmap.scaled(864, 576))
        self.tab_image5.setPixmap(new_pixmap1.scaled(1094, 576))

    def image_show(self):

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

    # 接收回傳的點並更新影像
    @QtCore.pyqtSlot(list)
    def get_return_points(self, points):
        print(points)
        self.return_points = points
        self.update_image()


class Analyze(QMainWindow, Ui_MainWindow, QtCore.QObject):
    returnPoints = QtCore.pyqtSignal(list)

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
        m_C, m_E, _ = CC_IQA.cc_task(self.rect_img, self.scale)
        self.label_C.setText("mean C: {:.4f}".format(m_C))
        self.label_E.setText("mean E: {:.4f}".format(m_E))
        pts = self.cc_image.return_points(self.ori_cc_img, self.get_p)
        if pts == False:
            QMessageBox.information(self, 'error', 'The number of selected points is insufficient',
                                    QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            return
        pts = list(map(tuple, pts))
        parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_file = os.path.join(parent_path, "points.txt")
        with open(output_file, 'w') as f:
            f.write(', '.join(str(p) for p in pts))
        self.returnPoints.emit(pts)
        self.close()

    def on_exit_clicked(self):
        QApplication.exit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    start_page = StartPage()
    start_page.show()
    sys.exit(app.exec_())

import os
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from ColorAnalysis import cba  # cba: ColorBlock Analysis

'''
if you want to add a new canvas, you can follow the steps below:
1. create a new class which extends the AnalyzeCanvas class
2. override the compute_initial_figure and update_figure methods
3. import the new class in the main.py file to use it
'''

class AnalyzeCanvas(FigureCanvas):
    """
    A custom canvas for displaying ocean data analysis results.

    Parameters:
    -----------
    parent : QWidget
        The parent widget of the canvas.
    width : float
        The width of the canvas in inches.
    height : float
        The height of the canvas in inches.
    dpi : int
        The resolution of the canvas in dots per inch.
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

    def compute_initial_figure(self):
        pass

    def update_figure(self, data):
        pass

class ColorBoardCanvas(AnalyzeCanvas):
    """
    A canvas for displaying color board data.

    This canvas extends the `AnalyzeCanvas` class and provides a way to display
    color board data using a sine wave plot. The `compute_initial_figure` method
    generates the plot using the NumPy library.

    Attributes:
        None
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100, num_subplots=3):
        super().__init__(parent, width, height, dpi)
        self.axes = [self.figure.add_subplot(1, num_subplots, i+1) for i in range(num_subplots)]
        self.compute_initial_figure()

    def compute_initial_figure(self):
        self.axes[0].set_title('k-means')
        self.axes[1].set_title('k-means_clean')
        self.axes[2].set_title('original')
        self.axes[0].text(0.5, 0.5, 'No', 
                        fontsize=40, ha='center',
                        va='center', color='red')
        self.axes[1].text(0.5, 0.5, 'Image',
                        fontsize=40, ha='center',
                        va='center', color='red')
        self.axes[2].text(0.5, 0.5, 'Uploaded',
                        fontsize=40, ha='center',
                        va='center', color='red')

    def update_figure(self, data):
        print('update_figure')
        self.axes[0].clear()
        self.axes[1].clear()
        self.axes[2].clear()
        self.axes[0].set_title('k-means')
        self.axes[1].set_title('k-means_clean')
        self.axes[2].set_title('original')
        center_rgb = data['center_rgb']
        center_rgb_clean = data['center_rgb_clean']
        rgb_list = data['rgb_list']
        # print(center_rgb)
        # print(center_rgb_clean)
        # print(rgb_list)

        for i in range(5):
            for j in range(5):
                if i == 4 and j == 4:
                    break
                x = j*0.2
                y = (4-i)*0.2
                # 由於 k-means 分群的結果是隨機的
                # 所以要將分群結果和色板的顏色進行比較
                self.axes[0].add_patch(plt.Rectangle((x, y), 0.2, 0.2, color=np.array(center_rgb[i * 5 + j]) / 255))
                self.axes[1].add_patch(plt.Rectangle((x, y), 0.2, 0.2, color=np.array(center_rgb_clean[i * 5 + j]) / 255))
                self.axes[2].add_patch(plt.Rectangle((x, y), 0.2, 0.2, color=np.array(rgb_list[i * 5 + j]) / 255))
                # 在方框中心位置添加 RGB 值文字
                rgb_text = f"({round(center_rgb[i * 5 + j][0])},{round(center_rgb[i * 5 + j][1])},{round(center_rgb[i * 5 + j][2])})"
                self.axes[0].text(x + 0.1, y + 0.1, rgb_text, color='black', fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
                rgb_clean_text = f"({round(center_rgb_clean[i * 5 + j][0])},{round(center_rgb_clean[i * 5 + j][1])},{round(center_rgb_clean[i * 5 + j][2])})"
                self.axes[1].text(x + 0.1, y + 0.1, rgb_clean_text, color='black', fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
                rgb_text = f"({round(rgb_list[i * 5 + j][0])},{round(rgb_list[i * 5 + j][1])},{round(rgb_list[i * 5 + j][2])})"
                self.axes[2].text(x + 0.1, y + 0.1, rgb_text, color='black', fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        self.draw()

class ColorBoardDeltaECanvas(AnalyzeCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100, num_subplots=3):
        super().__init__(parent, width, height, dpi)
        self.axes = [self.figure.add_subplot(1, num_subplots, i+1) for i in range(num_subplots)]
        self.compute_initial_figure()

    def set_subplot_properties(self, axes, title, color_indices, colordiff):
        axes.clear()
        axes.set_title(title)
        axes.bar(color_indices, colordiff)
        axes.set_xticks(color_indices)
        axes.set_xticklabels(color_indices)
        axes.set_xlabel('Color Index')
        axes.set_ylabel('Delta E')

    def compute_initial_figure(self):
        self.axes[0].set_title('Colors and Remove outlier Color')
        self.axes[1].set_title('Colors and Standard Color')
        self.axes[2].set_title('Remove outlier Colors and Standard Color')
        self.axes[0].text(0.5, 0.5, 'No', 
                        fontsize=40, ha='center',
                        va='center', color='red')
        self.axes[1].text(0.5, 0.5, 'Image',
                        fontsize=40, ha='center',
                        va='center', color='red')
        self.axes[2].text(0.5, 0.5, 'Uploaded',
                        fontsize=40, ha='center',
                        va='center', color='red')
    
    def update_figure(self, data):
        # 將 Delta E 值和顏色對應的索引轉換為 NumPy 陣列
        delta_e_rgb_rgbc = data['delta_e_rgb_rgbc']
        delta_e_rgb_std = data['delta_e_rgb_std']
        delta_e_rgbc_std = data['delta_e_rgbc_std']
        color_indices = data['color_indices']
        
        self.set_subplot_properties(self.axes[0], 'Colors and Remove outlier Color', color_indices, delta_e_rgb_rgbc)
        self.set_subplot_properties(self.axes[1], 'Colors and Standard Color', color_indices, delta_e_rgb_std)
        self.set_subplot_properties(self.axes[2], 'Remove outlier Colors and Standard Color', color_indices, delta_e_rgbc_std)

        self.axes[0].bar(color_indices, delta_e_rgb_rgbc, color='#2894FF')
        self.axes[1].bar(color_indices, delta_e_rgb_std, color='#2894FF')
        self.axes[2].bar(color_indices, delta_e_rgbc_std, color='#2894FF')

        self.draw()

class tanaAnalyze(AnalyzeCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100, num_subplots=1):
        super().__init__(parent, width, height, dpi)
        self.axes = [self.figure.add_subplot(1, num_subplots, i+1) for i in range(num_subplots)]
        self.compute_initial_figure()

    def compute_initial_figure(self):
        self.axes[0].text(0.5, 0.5, 'No Uploaded Image', 
                        fontsize=40, ha='center',
                        va='center', color='red')
        self.axes[0].set_title('Histogram of delta_e')
    
    def update_figure(self, data):
        center_rgb = data['center_rgb']
        rgb_list = data['rgb_list']
        self.axes[0].clear()
        #------------------cie_2000------------------
        #------- 色差比較圖 -------#
        center_rgb_temp = center_rgb.copy()
        center_rgb_temp = np.append(center_rgb_temp, [[0,0,0]], axis=0)
        # 改成5x5x3陣列
        center_rgb_2d = np.array(center_rgb_temp).reshape(5,5,3)
        # print(center_rgb_2d)
        rgb_list_float = np.array(rgb_list).reshape(5,5,3)
        # print(rgb_list_float)
        delta_e = cba.compare_colorboard(rgb_list_float, center_rgb_2d)
        plt.savefig(os.path.join('res', "delta_e.png")) # 儲存色差比較圖
        # fio.save_image_file('delta_e', 'res') # 如果要把所有生成的色差比較圖都留下，就用這個

        #------- 色差直方圖 -------#
        delta_e = np.array(delta_e)
        delta_e_mean = np.mean(delta_e[0:24]) # 去掉最後一個標記色塊
        # 繪製直方圖
        x = np.arange(0, 25)
        labels = [f"({i//5},{i%5})" for i in range(25)] 
        self.axes[0].bar(x, delta_e.reshape(25), width=0.4, label='delta_e: restored_model')
        # 繪製平均值
        self.axes[0].axhline(delta_e_mean, color='r', linestyle='--', label='delta_e mean')
        self.axes[0].text(24.6, delta_e_mean, f"{delta_e_mean:.2f}", ha='right', va='bottom')
        # 設定圖表屬性
        self.axes[0].set_xticks(x+0.4/2)
        self.axes[0].set_xticklabels(labels)
        self.axes[0].legend()
        self.axes[0].set_title('Histogram of delta_e')
        self.axes[0].set_xlabel('(i,j)')
        self.axes[0].set_ylabel('Value')

        self.draw()
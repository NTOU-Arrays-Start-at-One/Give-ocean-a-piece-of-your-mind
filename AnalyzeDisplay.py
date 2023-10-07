from matplotlib import pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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
        self.axes1 = fig.add_subplot(131)
        self.axes2 = fig.add_subplot(132)
        self.axes3 = fig.add_subplot(133)

        self.compute_initial_figure()

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
    def compute_initial_figure(self):
        self.axes1.set_title('k-means')
        self.axes2.set_title('k-means_clean')
        self.axes3.set_title('original')
        self.axes1.text(0.5, 0.5, 'No', 
                        fontsize=40, ha='center',
                        va='center', color='red')
        self.axes2.text(0.5, 0.5, 'Image',
                        fontsize=40, ha='center',
                        va='center', color='red')
        self.axes3.text(0.5, 0.5, 'Uploaded',
                        fontsize=40, ha='center',
                        va='center', color='red')

    def update_figure(self, data):
        print('update_figure')
        self.axes1.clear()
        self.axes2.clear()
        self.axes3.clear()
        self.axes1.set_title('k-means')
        self.axes2.set_title('k-means_clean')
        self.axes3.set_title('original')
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
                self.axes1.add_patch(plt.Rectangle((x, y), 0.2, 0.2, color=np.array(center_rgb[i * 5 + j]) / 255))
                self.axes2.add_patch(plt.Rectangle((x, y), 0.2, 0.2, color=np.array(center_rgb_clean[i * 5 + j]) / 255))
                self.axes3.add_patch(plt.Rectangle((x, y), 0.2, 0.2, color=np.array(rgb_list[i * 5 + j]) / 255))
                # 在方框中心位置添加 RGB 值文字
                rgb_text = f"({round(center_rgb[i * 5 + j][0])},{round(center_rgb[i * 5 + j][1])},{round(center_rgb[i * 5 + j][2])})"
                self.axes1.text(x + 0.1, y + 0.1, rgb_text, color='black', fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
                rgb_clean_text = f"({round(center_rgb_clean[i * 5 + j][0])},{round(center_rgb_clean[i * 5 + j][1])},{round(center_rgb_clean[i * 5 + j][2])})"
                self.axes2.text(x + 0.1, y + 0.1, rgb_clean_text, color='black', fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
                rgb_text = f"({round(rgb_list[i * 5 + j][0])},{round(rgb_list[i * 5 + j][1])},{round(rgb_list[i * 5 + j][2])})"
                self.axes3.text(x + 0.1, y + 0.1, rgb_text, color='black', fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        plt.tight_layout()
        self.draw()

class ColorBoardDeltaECanvas(AnalyzeCanvas):

    def set_subplot_properties(self, axes, title, color_indices, colordiff):
        axes.clear()
        axes.set_title(title)
        axes.bar(color_indices, colordiff)
        axes.set_xticks(color_indices)
        axes.set_xticklabels(color_indices)
        axes.set_xlabel('Color Index')
        axes.set_ylabel('Delta E')

    def compute_initial_figure(self):
        self.axes1.set_title('Colors and Remove outlier Color')
        self.axes2.set_title('k-means_clean')
        self.axes3.set_title('original')
        self.axes1.text(0.5, 0.5, 'No', 
                        fontsize=40, ha='center',
                        va='center', color='red')
        self.axes2.text(0.5, 0.5, 'Image',
                        fontsize=40, ha='center',
                        va='center', color='red')
        self.axes3.text(0.5, 0.5, 'Uploaded',
                        fontsize=40, ha='center',
                        va='center', color='red')
    
    def update_figure(self, data):
        # 將 Delta E 值和顏色對應的索引轉換為 NumPy 陣列
        delta_e_rgb_rgbc = data['delta_e_rgb_rgbc']
        delta_e_rgb_std = data['delta_e_rgb_std']
        delta_e_rgbc_std = data['delta_e_rgbc_std']
        color_indices = data['color_indices']
        
        self.set_subplot_properties(self.axes1, 'Colors and Remove outlier Color', color_indices, delta_e_rgb_rgbc)
        self.set_subplot_properties(self.axes2, 'Colors and Standard Color', color_indices, delta_e_rgb_std)
        self.set_subplot_properties(self.axes3, 'Remove outlier Colors and Standard Color', color_indices, delta_e_rgbc_std)
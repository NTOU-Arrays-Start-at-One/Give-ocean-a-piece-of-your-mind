import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # 表格顏色
from ColorAnalysis import cba  # cba: ColorBlock Analysis
from ColorAnalysis import fileio as fio # to save file
from ColorAnalysis import GUI # GUI: Graphical User Interface to show image

def colorAnalysis(crop_image):

    #-------------------------------------------------------------------------#
    # 圖片的空間轉換
    #-------------------------------------------------------------------------#

    # Convert PIL image to numpy array
    np_array = np.array(crop_image)

    # Convert numpy array to OpenCV image (BGR format)
    crop_image = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)

    #-------------------------------------------------------------------------#
    # 常數設定
    #-------------------------------------------------------------------------#

    result_dir = fio.find_result_dir()
    #-------------------------------------------------------------------------#
    # 色板影像校正
    #-------------------------------------------------------------------------#

    # 讀入讀檔
    standard_val, standard_unwarp = cba.correction_and_analysis(cv2.imread("src/Standard.png"), False)
    restored_val, restored_unwarp = cba.correction_and_analysis(crop_image, False)

    #-------------------------------------------------------------------------#
    # 比較兩色版的差異
    #-------------------------------------------------------------------------#
    # 顯示校正後的色板
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(cv2.cvtColor(standard_unwarp, cv2.COLOR_BGR2RGB))
    ax1.set_title('Standard')
    ax2.imshow(cv2.cvtColor(restored_unwarp, cv2.COLOR_BGR2RGB))
    ax2.set_title('Restored')
    fio.save_image_file('delta_e_1_unwarp_restored_model', result_dir)

    # 測試與比較
    # 造出色差比較圖
    delta_e_1 = cba.compare_colorboard(standard_val, restored_val) # 造出色差比較圖
    fio.save_image_file('delta_e_1', result_dir) # 儲存色差比較圖
    fio.save_text_file(delta_e_1, 'delta_e_1', result_dir) # 儲存excel文字紀錄
    plt.show() # 將兩張圖一同顯示

    # 造出色彩通道比較圖
    diff_1 = cba.color_diff_colorboard(standard_val, restored_val) # 造出色差比較圖
    # fio.save_image_file('diff_1', result_dir) # 儲存色差比較圖
    # fio.save_text_file(diff_1, 'diff_1', result_dir) # 儲存excel文字紀錄
    plt.show() # 將兩張圖一同顯示

    #-------------------------------------------------------------------------#
    # 顯示兩色版的差異
    #-------------------------------------------------------------------------#
    # 顯示
    f, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.imshow(cv2.cvtColor(standard_unwarp, cv2.COLOR_BGR2RGB))
    ax1.set_title('Standard')
    ax3.imshow(cv2.cvtColor(restored_unwarp, cv2.COLOR_BGR2RGB))
    ax3.set_title('restored_model')
    fio.save_image_file('unwarp_restored_model and Standard_image', result_dir)

    # 計算平均值
    delta_e_1 = np.array(delta_e_1)
    delta_e_1_mean = np.mean(delta_e_1) # restored model

    # 繪製直方圖
    x = np.arange(0, 25)
    labels = [f"({i//5},{i%5})" for i in range(25)]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, delta_e_1.reshape(25), width=0.4, label='delta_e: restored_model')

    # 繪製平均值
    ax.axhline(delta_e_1_mean, color='r', linestyle='--', label='delta_e mean')

    ax.text(24.6, delta_e_1_mean, f"{delta_e_1_mean:.2f}", ha='right', va='bottom')

    # 設定圖表屬性
    ax.set_xticks(x+0.4/2)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title('Histogram of delta_e')
    ax.set_xlabel('(i,j)')
    ax.set_ylabel('Value')

    fio.save_image_file('Histogram of delta_e', result_dir)
    plt.show()

    #-------------------------------------------------------------------------#
    # 顯示兩色版的差異
    #-------------------------------------------------------------------------#
    # 顯示圖片差異介面
    GUI.imageCompare(standard_unwarp, restored_unwarp)
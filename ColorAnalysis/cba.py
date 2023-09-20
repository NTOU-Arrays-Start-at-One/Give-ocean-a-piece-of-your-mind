#----------------------------------------------------#
#   cba.py: ColorBlock Analysis
#   用於將色板內的色塊提取出來並檢驗顏色
#----------------------------------------------------#

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ColorAnalysis import cbe  # cbe: ColorBoard Extraction
from ColorAnalysis import fileio as fio # to save file

from skimage import io
from skimage import transform
from skimage.metrics import structural_similarity as ssim # ssim 相似比較

# 正規化
from sklearn.preprocessing import MinMaxScaler 

# 色差公式
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie1976, delta_e_cie2000

#----------------------------------------------------#
# draw_rect 
# 顯示每一個測試色塊的範圍
# 以100pixel為單位劃分為紅色表格，並向內縮特定大小為最終的測試色塊，以減少校正後的誤差。 
#----------------------------------------------------#
def draw_rect(img):
  im = img.copy()
  f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
  
  # 繪製水平線
  for i in range(0, im.shape[0], 100):
    cv2.line(im, (0, i), (im.shape[1], i), (0, 0, 255), 2)

  # 繪製垂直線
  for j in range(0, im.shape[1], 100):
    cv2.line(im, (j, 0), (j, im.shape[0]), (0, 0, 255), 2)
  
  ax1.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
  ax1.set_title('Grid')
  
  # 繪製矩形
  for i in range(0, im.shape[0], 100):
    for j in range(0, im.shape[1], 100):
      # 矩形的左上角座標
      x1 = j + 10
      y1 = i + 10
      # 矩形的右下角座標
      x2 = j + 90
      y2 = i + 90
      # 畫矩形
      cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
  ax2.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
  ax2.set_title('Block')

  # 顯示圖像
  plt.show()

#----------------------------------------------------#
# color_analysis
# 切割出每一個測試色塊的範圍，並使用高斯濾波器後計算每一色塊的均值，以表示此色塊的代表色。
#----------------------------------------------------#
def color_analysis(im):
  colorBlockImage = []
  for i in range(0, im.shape[0]-2): # -2是為了扣掉邊框
    row = []
    for j in range(0, im.shape[1]-2):
      # 定義擷取區域的左上角和右下角座標
      x1, y1 = 110 + j*100, 110 + i*100 # 色塊向內縮10pixel
      x2, y2 = 190 + j*100, 190 + i*100
      row.append(im[y1:y2+1, x1:x2+1])
    colorBlockImage.append(row)
  
  colorBlockVal = []
  for i in range(0, 5):
    row = []
    for j in range(0, 5):
      # 使用高斯濾波器，去除躁點
      blurred = cv2.GaussianBlur(colorBlockImage[i][j], (5, 5), 0)

      # 計算均值
      val_mean = cv2.mean(blurred)
      row.append(val_mean)

    colorBlockVal.append(row)
  return colorBlockVal

#----------------------------------------------------#
# get_delta_e
# 以CIE1976和CIEDE2000計算測試色塊間的色差值，用來比較色彩還原效果。
#----------------------------------------------------#
def get_delta_e(color1, color2):
  # 色彩空間轉換(BGR to LAB)  
  def rgb2lab(rgb):
    # print(f"Input RGB: {rgb}")
    return convert_color(sRGBColor(rgb[0], rgb[1], rgb[2]), LabColor)


  # 計算CIE1976和CIEDE2000的色差
  delta_e_1976 = delta_e_cie1976(rgb2lab(color1), rgb2lab(color2))
  delta_e_2000 = delta_e_cie2000(rgb2lab(color1), rgb2lab(color2))

  return delta_e_2000

#----------------------------------------------------#
# get_color_channel_diff_percent
# 比較兩色塊間的色彩三通道差百分比，用來比較色彩還原效果。
#----------------------------------------------------#
def get_color_channel_diff_percent(color1, color2):
    r1, g1, b1, a1 = color1
    r2, g2, b2, a2 = color2
    delta_r = abs(r2 - r1)
    delta_g = abs(g2 - g1)
    delta_b = abs(b2 - b1)
    return delta_r / 255 * 100, delta_g / 255 * 100, delta_b / 255 * 100
  
#----------------------------------------------------#
# get_ssim_score
# 以structural similarity index，SSIM index計算測試色塊間的相似程度，用來比較色彩還原效果。
#----------------------------------------------------#
def get_ssim_score(im1, im2):
  gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # 計算兩個灰度圖像之間的SSIM
  score, diff = ssim(gray1, gray2, full=True)
  return score

#----------------------------------------------------#
# correction_and_analysis
# 對圖板做色塊的影像處理與分析
#----------------------------------------------------#
def correction_and_analysis(colorBoard, display = 1):
  colorBoard_copy = colorBoard.copy() # 複製副本以避免畫圖影響結果
  rect_point = cbe.find_edge_of_colorboard(colorBoard, False) # 找尋邊緣，並回傳矩形的頂點座標
  dc_img = cbe.perspective_correction(colorBoard_copy, rect_point, False) # 對矩形的頂點座標做透視校正
  dc_img = cbe.rotate(dc_img) # 旋轉
  colorBlockVal = color_analysis(dc_img) # 處理並輸出色塊的代表顏色，colorBlockVal[i][j]為色塊顏色
  #draw_rect(dc_img) # 畫出色塊

  if display == 1:
    plt.imshow(cv2.cvtColor(dc_img, cv2.COLOR_BGR2RGB))
    # 代表顏色的表格輸出
    fig, ax = plt.subplots()
    ax.axis('off')
    # 設置單元格文本和顏色
    cell_text = []
    for i in range(5):
        row_text = []
        row_colors = []
        for j in range(5):
            # 將RGB顏色塊和像素值一起顯示
            cell_val = f"({i},{j})\n\n{colorBlockVal[i][j][0]:.1f}\n{colorBlockVal[i][j][1]:.1f}\n{colorBlockVal[i][j][2]:.1f}"
            row_text.append(cell_val)
        cell_text.append(row_text)
    # 創建表格
    table = ax.table(cellText=cell_text, cellLoc='center', bbox=[0,0,1,1])
    # 設置表格標題
    ax.set_title('colorBlockVal')
    # 設置表格大小和字體大小
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)
    # 設置圖形大小
    fig.set_figwidth(8)
    fig.set_figheight(8)

    plt.show()

  return colorBlockVal, dc_img # colorBlockVal: 輸出色板上每色塊的代表色, dc_img: 透視校正後的圖

#----------------------------------------------------#
# compare_colorboard
# 對兩色板做色差比較
#----------------------------------------------------#
def compare_colorboard(a_val, b_val): # 色差計算
  # 計算色差
  delta_e = []
  for i in range(0, 5):
    row = []
    for j in range(0, 5):
      row.append(get_delta_e(a_val[i][j],b_val[i][j]))
    delta_e.append(row)
  # 將 delta_e 正規化到 0~1 範圍內
  scaler = MinMaxScaler()
  delta_e_norm = scaler.fit_transform(delta_e)
  # 表格輸出
  fig, ax = plt.subplots()
  ax.axis('off')
  # 設定表格內容
  cell_text = []
  for i in range(5):
      row = []
      for j in range(5):
          row.append(f"({i},{j}),{delta_e[i][j]:.3f}")
      cell_text.append(row)
  
  # 繪製表格並設定顏色
  table = ax.table(cellText=cell_text, cellLoc='center', bbox=[0,0,0.8,1], cellColours=plt.cm.Greens(delta_e_norm))
  ax.set_title('Delta E')
  ax.set_aspect('equal')
  table.auto_set_font_size(False)
  table.set_fontsize(14)
  # 設定表格大小和位置
  table.scale(1, 2)
  # 設定長和寬大小
  fig.set_figwidth(10)
  fig.set_figheight(10)
  # 增加 color bar 對照
  cax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
  sm = plt.cm.ScalarMappable(cmap=plt.cm.Greens, norm=plt.Normalize(vmin=0, vmax=1))
  sm.set_array([])
  fig.colorbar(sm, cax=cax)

  delta_e_copy = delta_e.copy()
  return delta_e_copy

#----------------------------------------------------#
# color_diff_colorboard
# 對兩色板做色彩通道比較
#----------------------------------------------------#
def color_diff_colorboard(a_val, b_val):
  # 計算通道差異
  diff = []
  for i in range(0, 5):
    row = []
    for j in range(0, 5):
      row.append(get_color_channel_diff_percent(a_val[i][j],b_val[i][j]))
    diff.append(row)

  # 代表顏色的表格輸出
  fig, ax = plt.subplots()
  ax.axis('off')
  # 設置單元格文本和顏色
  cell_text = []
  for i in range(5):
      row_text = []
      row_colors = []
      for j in range(5):
          # 將RGB顏色塊和像素值一起顯示
          cell_val = f"({i},{j})\n\n{diff[i][j][0]:.1f}%\n{diff[i][j][1]:.1f}%\n{diff[i][j][2]:.1f}%"
          row_text.append(cell_val)
      cell_text.append(row_text)
  # 創建表格
  table = ax.table(cellText=cell_text, cellLoc='center', bbox=[0,0,1,1])
  # 設置表格標題
  ax.set_title('colorBlockVal')
  # 設置表格大小和字體大小
  table.auto_set_font_size(False)
  table.set_fontsize(14)
  table.scale(1, 2)
  # 設置圖形大小
  fig.set_figwidth(8)
  fig.set_figheight(8)
  plt.show()

  diff_copy = diff.copy()
  return diff_copy
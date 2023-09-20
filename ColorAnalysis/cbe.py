#----------------------------------------------------#
#   cbe.py: ColorBoard Extraction
#   用於將色板提取出來測試
#----------------------------------------------------#

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------邊緣檢測--------------------------------- #

#----------------------------------------------------#
#   find_edge_of_colorboard
#   尋找色板邊緣，用於透視校正。
#----------------------------------------------------#
def find_edge_of_colorboard(im, display = 1):
    # cv2.imshow(im)

    # 圖片轉為灰階
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2.imshow(img_gray)

    # 高斯濾波
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # 將圖片進行邊緣檢測
    edges = cv2.Canny(blurred, 110, 250)

    # 尋找輪廓
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 遍歷所有輪廓
    rect_points = []  # 紀錄矩形的頂點
    for contour in contours:
        # 將輪廓近似成矩形
        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)

        # 如果矩形的頂點數為4，就畫出矩形
        if len(approx) == 4:
            cv2.drawContours(im, [approx], 0, (0, 255, 0), 2)
            point = []
            for point in approx:
                rect_points.append((point[0][0], point[0][1]))
                cv2.circle(im, (point[0][0], point[0][1]), 5, (0, 0, 255), -1)
                
    # 是否顯示結果
    if display == 1:
      cv2.imshow('find_corners',im)

    return rect_points

# ---------------------------------透視校正--------------------------------- #

#----------------------------------------------------#
#   unwarp
#   影像映射、展平。
#----------------------------------------------------#
def unwarp(img, src, dst, display = 1):
    h, w = img.shape[:2]
    dstW, dstH = map(int, dst[3])
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (dstW, dstH), flags=cv2.INTER_LINEAR)

    if display:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        f.subplots_adjust(hspace=.2, wspace=.05)
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
        y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
        ax1.plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
        ax1.set_ylim([h, 0])
        ax1.set_xlim([0, w])
        ax1.set_title('Original Image', fontsize=15)
        #ax2.imshow(cv2.flip(warped, 1)) #1:水平翻轉 0:垂直翻轉 -1:水平垂直翻轉
        ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        ax2.set_title('Distortion Correction Result', fontsize=15)
        return warped
    else:
        return warped

#----------------------------------------------------#
#   perspective_correction
#   對色板透視校正，將色板抓出。
#----------------------------------------------------#
def perspective_correction(im, rect_point, display = 1):
  #圖片大小
  w, h = im.shape[0], im.shape[1]
  
  # 頂點(idx由色板右上開始逆時鐘編號)
  # for i in range(0,len(rect_point)): 
  #   if(i==4): print()
  #   print(rect_point[i], end='')
  
  # bug: 可能會因為點的順序錯誤導致透視校正有問題
  # fixed：讓座標能夠(左上，右上，左下，右下)
  # rect_point = sorted(rect_point, key=lambda x: x[0])
  # rect_point = sorted(rect_point, key=lambda x: x[1])
  # rect_point = sorted(rect_point, key=lambda x: x[0], reverse=True)
  # rect_point = sorted(rect_point, key=lambda x: x[1], reverse=True)

  # 座標(左上，右上，左下，右下)
  src = np.float32([(rect_point[1]),
          (rect_point[0]),
          (rect_point[2]),
          (rect_point[3])])

  dst = np.float32([(0, 0),
          (700, 0),
          (0, 700),
          (700, 700)])

  #校正與輸出
  return unwarp(im, src, dst, display)

# ---------------------------------色板方向校正--------------------------------- #

#----------------------------------------------------#
#   find_rotation_marker
#   利用標準差尋找旋轉用的標記
#----------------------------------------------------#
def find_rotation_marker(img):
  im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  colorBlockImage = []
  for i in range(0, 2):
    row = []
    for j in range(0, 2):
      # 定義擷取區域的左上角和右下角座標
      x1, y1 = 110 + j*400, 110 + i*400 # 色塊向內縮10pixel
      x2, y2 = 190 + j*400, 190 + i*400
      row.append(im[y1:y2+1, x1:x2+1])
    colorBlockImage.append(row)
  
  colorBlockStd = []
  for i in range(0, 2):
    row = []
    for j in range(0, 2):
      # 計算標準差
      val_std = np.std(colorBlockImage[i][j])
      colorBlockStd.append({"idx": (i, j), "val_std": val_std})

  # print(colorBlockStd)
  # 找出標準差最大(均勻度最小)的那一個
  max_std = max(colorBlockStd, key=lambda x: x["val_std"])
  # print(max_std["idx"])
  return max_std["idx"] 

#----------------------------------------------------#
#   rotate
#   旋轉色板至正確方向
#----------------------------------------------------#
def rotate(colorBoard):
  mark = find_rotation_marker(colorBoard)
  if mark == (0,0):
    angle = 90
  elif mark == (0,1):
    angle = 180
  elif mark == (1,1):
    angle = 270
  else: angle = 0

  (h, w, d) = colorBoard.shape
  center = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D(center, angle, 1.0)
  colorBoard = cv2.warpAffine(colorBoard, M, (w, h))
  return colorBoard
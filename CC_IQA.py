import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
from colormath.color_diff import delta_e_cie2000

####  The data from https://www.xrite.com/service-support/new_color_specifications_for_colorchecker_sg_and_classic_charts
CIE_lab_24 = [[37.54, 14.37, 14.92], [64.66, 19.27, 17.5], [49.32, -3.82, -22.54], [43.46, -12.74, 22.72], [54.94, 9.61, -24.79], [70.48, -32.26, -0.37],
              [62.73, 35.83, 56.5], [39.43, 10.75, -45.17], [50.57, 48.64, 16.67], [30.1, 22.54, -20.87], [71.77, -24.13, 58.19], [71.51, 18.24, 67.37],
              [28.37, 15.42, -49.8], [54.38, -39.72, 32.27], [42.43, 51.05, 28.62], [81.8, 2.67, 80.41], [50.63, 51.28, -14.12], [49.57, -29.71, -28.32],
              [95.19, -1.03, 2.93], [81.29, -0.57, 0.44], [66.89, -0.75, -0.06], [50.76, -0.13, 0.14], [35.63, -0.46, -0.48], [20.64, 0.07, -0.46]]
def rgb2lab(rgb):
    r = rgb[0] / 255.0  # rgb range: 0 ~ 1
    g = rgb[1] / 255.0
    b = rgb[2] / 255.0
    # gamma 2.2
    if r > 0.04045:
        r = pow((r + 0.055) / 1.055, 2.4)
    else:
        r = r / 12.92
    if g > 0.04045:
        g = pow((g + 0.055) / 1.055, 2.4)
    else:
        g = g / 12.92
    if b > 0.04045:
        b = pow((b + 0.055) / 1.055, 2.4)
    else:
        b = b / 12.92
    # sRGB
    X = r * 0.436052025 + g * 0.385081593 + b * 0.143087414
    Y = r * 0.222491598 + g * 0.716886060 + b * 0.060621486
    Z = r * 0.013929122 + g * 0.097097002 + b * 0.714185470
    # XYZ range: 0~100
    X = X * 100.000
    Y = Y * 100.000
    Z = Z * 100.000
    # Reference White Point
    ref_X = 96.4221
    ref_Y = 100.000
    ref_Z = 82.5211
    X = X / ref_X
    Y = Y / ref_Y
    Z = Z / ref_Z
    # Lab
    if X > 0.008856:
        X = pow(X, 1 / 3.000)
    else:
        X = (7.787 * X) + (16 / 116.000)
    if Y > 0.008856:
        Y = pow(Y, 1 / 3.000)
    else:
        Y = (7.787 * Y) + (16 / 116.000)
    if Z > 0.008856:
        Z = pow(Z, 1 / 3.000)
    else:
        Z = (7.787 * Z) + (16 / 116.000)
    Lab_L = round((116.000 * Y) - 16.000, 2)
    Lab_a = round(500.000 * (X - Y), 2)
    Lab_b = round(200.000 * (Y - Z), 2)
    return Lab_L, Lab_a, Lab_b


def color_difference(measure, ref):
    L = abs(measure[0] - ref[0])
    a = abs(measure[1] - ref[1])
    b = abs(measure[2] - ref[2])
    C = (a ** 2 + b ** 2) ** 0.5
    E = (a ** 2 + b ** 2 + L ** 2) ** 0.5
    return C, E


def draw_rect(img, lt, rb):
    point_color = (0, 255, 0)  # BGR
    thickness = 2
    lineType = 4
    result = cv2.rectangle(img, lt, rb, point_color, thickness, lineType)
    return result

def load_std_ref_LAB(c_type):
    global REF_LAB
    REF_LAB = []
    if c_type == "24":
        REF_LAB = CIE_lab_24
    else:
        assert True, "type error"

def calculate_delta_e(color1, color2):
    # 將 RGB 轉換為 Lab 色彩空間
    lab_color1 = rgb_to_lab(color1)
    lab_color2 = rgb_to_lab(color2)
    
    # 計算 Delta E
    delta_e = distance.euclidean(lab_color1, lab_color2)
    
    return delta_e

def rgb_to_lab(rgb_color):
    # 將 RGB 色彩空間轉換為 XYZ 色彩空間
    xyz_color = rgb_to_xyz(rgb_color)
    
    # 將 XYZ 色彩空間轉換為 Lab 色彩空間
    lab_color = xyz_to_lab(xyz_color)
    
    return lab_color

def rgb_to_xyz(rgb_color):
    # 將 RGB 轉換為 XYZ 色彩空間的矩陣
    rgb_to_xyz_matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                                  [0.2126729, 0.7151522, 0.0721750],
                                  [0.0193339, 0.1191920, 0.9503041]])
    
    # 將 RGB 值歸一化為 0-1 的範圍
    rgb_color = np.array(rgb_color) / 255.0
    
    # 將 gamma 矯正應用於 RGB 值
    rgb_color = np.where(rgb_color <= 0.04045, rgb_color / 12.92, ((rgb_color + 0.055) / 1.055) ** 2.4)
    
    # 將 gamma 矯正後的 RGB 值與轉換矩陣相乘，得到 XYZ 值
    xyz_color = np.dot(rgb_to_xyz_matrix, rgb_color)
    
    return xyz_color

def xyz_to_lab(xyz_color):
    # 定義白點的 XYZ 值
    reference_white = [0.95047, 1.00000, 1.08883]
    
    # 歸一化 XYZ 值除以白點的 XYZ 值，得到 XYZ 的百分比
    xyz_percentage = xyz_color / reference_white
    
    # 定義 Lab 轉換的函數
    def lab_func(t):
        return np.where(t > (6 / 29) ** 3, t ** (1 / 3), (1 / 3) * ((29 / 6) ** 2) * t + 4 / 29)
    
    # 將 XYZ 的百分比轉換為 Lab 值
    lab_color = np.array([116 * lab_func(xyz_percentage[1]) - 16,
                          500 * (lab_func(xyz_percentage[0]) - lab_func(xyz_percentage[1])),
                          200 * (lab_func(xyz_percentage[1]) - lab_func(xyz_percentage[2]))])
    
    return lab_color

def remove_outliers(rgb_array, threshold):
    rgb_np = np.array(rgb_array)

    #print("rgb_np[0]")
    #print(rgb_np[:,:,0])
    #print("rgb_np[1]")
    #print(rgb_np[:,:,1])
    #print("rgb_np[2]")
    #print(rgb_np[:,:,2])
    # 找到每個通道的最小值和最大值
    min_vals_r = np.min(rgb_np[:,:,0])
    min_vals_g = np.min(rgb_np[:,:,1])
    min_vals_b = np.min(rgb_np[:,:,2])
    max_vals_r = np.max(rgb_np[:,:,0])
    max_vals_g = np.max(rgb_np[:,:,1])
    max_vals_b = np.max(rgb_np[:,:,2])

    print(f"min_vals_r:{min_vals_r}")
    print(f"min_vals_g:{min_vals_g}")
    print(f"min_vals_b:{min_vals_b}")
    print(f"max_vals_r:{max_vals_r}")
    print(f"max_vals_g:{max_vals_g}")
    print(f"max_vals_b:{max_vals_b}")

    # 判斷是否為極值
    is_outlier_r = np.logical_or(rgb_np[:,:,0] < min_vals_r + threshold, rgb_np[:,:,0] > max_vals_r - threshold)
    #print("is_outlier_r:")
    #print(is_outlier_r)
    is_outlier_g = np.logical_or(rgb_np[:,:,1] < min_vals_g + threshold, rgb_np[:,:,1] > max_vals_g - threshold)
    #print("is_outlier_g:")
    #print(is_outlier_g)
    is_outlier_b = np.logical_or(rgb_np[:,:,2] < min_vals_b + threshold, rgb_np[:,:,2] > max_vals_b - threshold)
    #print("is_outlier_b:")
    #print(is_outlier_b)

    # 組合三個通道的極值判斷結果
    is_outlier = np.logical_or.reduce([is_outlier_r, is_outlier_g, is_outlier_b])
    # 返回篩選後的 RGB 值陣列
    filtered_rgb = rgb_np[~is_outlier]
    #print("filtered_rgb:")
    #print(filtered_rgb)
    return filtered_rgb

def cc_task(cc_img, scale=0.5):
    #####  The data from https://www.xrite.com/service-support/new_color_specifications_for_colorchecker_sg_and_classic_charts
    load_std_ref_LAB("24")
    h, w = cc_img.shape[0], cc_img.shape[1]

    # 色板 RGB 值的列表
    # 由右到左，由上到下顯示
    rgb_list = [[51, 51, 51], [102, 102, 102], [153, 153, 153], [204, 204, 204], [255, 255, 255],
                [0, 0, 51], [0, 0, 102], [0, 0, 153], [0, 0, 204], [0, 0, 255],
                [0, 128, 255], [128, 255, 0], [255, 255, 0], [255, 190, 0], [255, 0, 0],
                [0, 51, 0], [0, 102, 0], [0, 153, 0], [0, 204, 0], [0, 255, 0],
                [128, 74, 0], [131, 0, 181], [255, 0, 255], [0, 255, 255], [0, 0, 0]]

    h_block = (h / 7)
    w_block = (w / 7)
    center_points = []
    real_center_points = []
    for y in range(1, 7):
        for x in range(1, 7):
            center_points.append([x * w_block, y * h_block])
    for i in range(6):
        for j in range(6):
            if i == 0 or j == 0:
                continue
            real_center_points.append(center_points[i * 6 + j])
    center_points = real_center_points[0:24]
    #print("center_points", center_points)
    img_rect = cc_img
    img_rect_ = img_rect.copy()
    count = 0
    mean_C = 0
    mean_E = 0
    center_rgb = []
    center_rgb_clean = []
    # rgb跟rgbc的差異
    rgb_rgbc_cmp = []
    # rgb跟原色板的差異
    rgb_std_cmp = []
    # rgbc跟原色板的差異
    rgbc_std_cmp = []
    for c in center_points:
        # 解釋變數
        # c: 中心點座標
        # 中心點座標是指每個色塊的中心點
        # 每個色塊的中心點是怎麼找的呢?
        # 以 24 格的 ColorChecker 為例
        # 24 格的 ColorChecker 是 4 * 6 的矩陣
        # 每個色塊的寬度是整張圖片的 1/6
        # 每個色塊的高度是整張圖片的 1/4
        # 所以每個色塊的中心點座標是
        # (1/6 * 1/2, 1/4 * 1/2) ~ (5/6 * 1/2, 3/4 * 1/2)
        # c_w: 中心點寬度
        # c_h: 中心點高度
        # lt: 左上角座標
        # rb: 右下角座標
        # cc_block: 中心點區域
        c_w = c[0] - w_block / 2
        c_h = c[1] - h_block / 2
        lt = (int(c_w - (w_block / 2) * scale), int(c_h - (h_block / 2) * scale))
        rb = (int(c_w + (w_block / 2) * scale), int(c_h + (h_block / 2) * scale))

        img_rect_ = draw_rect(img_rect_, lt, rb)
        print("lt, rb", lt, rb)
        cc_block = img_rect[lt[1]: rb[1], lt[0]: rb[0]]
        print("cc_block", cc_block)
        cc_block_clean = cc_block.copy()
        cc_block_clean = remove_outliers(cc_block_clean, 1)
        R_mean = np.mean(cc_block[:, :, 0])
        G_mean = np.mean(cc_block[:, :, 1])
        B_mean = np.mean(cc_block[:, :, 2])
        R_mean_clean = np.mean(cc_block_clean[:, 0])
        G_mean_clean = np.mean(cc_block_clean[:, 1])
        B_mean_clean = np.mean(cc_block_clean[:, 2])
        # RGB_m: 中心點 RGB 值
        RGB_m = [R_mean, G_mean, B_mean]
        RGB_m_clean = [R_mean_clean, G_mean_clean, B_mean_clean]
        center_rgb.append(RGB_m)
        center_rgb_clean.append(RGB_m_clean)
        rgb_rgbc_cmp.append(calculate_delta_e(RGB_m, RGB_m_clean))
        rgb_std_cmp.append(calculate_delta_e(RGB_m, rgb_list[count]))
        rgbc_std_cmp.append(calculate_delta_e(RGB_m_clean, rgb_list[count]))
        # LAB_m: 中心點 LAB 值
        LAB_m = rgb2lab(RGB_m)
        #print(f"count: {count}")
        #print("LAB_m: ", LAB_m, "REF_LAB: ", REF_LAB[count])

        C, E = color_difference(LAB_m, REF_LAB[count])

        mean_C += C
        mean_E += E

        count += 1
    
    # 由於中心點的 RGB 值會受到環境光影響
    # 所以我們需要將 RGB 值濾掉離群值
    # 這裡我們使用的方法是
    # 將 RGB 值轉換成 LAB 值
    # 再將 LAB 值轉換成 XYZ 值
    # 再將 XYZ 值轉換成 LAB 值
    # 再將 LAB 值轉換成 RGB 值
    # 這樣就可以將 RGB 值濾掉離群值
    # 這裡我們使用的是 OpenCV 的函式
    # 這裡的函式會將 RGB 值轉換成 XYZ 值
    # 再將 XYZ 值轉換成 LAB 值
    # 再將 LAB 值轉換成 RGB 值

    #------------------k-means------------------
    X = np.array(center_rgb)
    # k-means 分群
    kmeans = KMeans(n_clusters=5).fit(X)
    # 每個色塊所屬的群組
    labels = kmeans.labels_
    # 創建圖表
    grid = plt.GridSpec(6, 6)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0, 0].set_title('k-means')
    axs[0, 1].set_title('k-means_clean')
    axs[0, 2].set_title('original')
    for i in range(5):
        for j in range(5):
            if i == 4 and j == 4:
                break
            x = j*4
            y = (5-i)*4
            # 由於 k-means 分群的結果是隨機的
            # 所以要將分群結果和色板的顏色進行比較
            axs[0, 0].add_patch(plt.Rectangle((x, y), 4, 4, color=np.array(center_rgb[i * 5 + j]) / 255))
            axs[0, 1].add_patch(plt.Rectangle((x, y), 4, 4, color=np.array(center_rgb_clean[i * 5 + j]) / 255))
            axs[0, 2].add_patch(plt.Rectangle((x, y), 4, 4, color=np.array(rgb_list[i * 5 + j]) / 255))
    axs[0, 0].axis('scaled')
    axs[0, 1].axis('scaled')
    axs[0, 2].axis('scaled')

    #------------------k-means------------------
    #------------------每一點的rgb值與色差------------------
    axs[1, 0].set_title('k-means')
    axs[1, 1].set_title('k-means_clean')
    axs[1, 2].set_title('original')
    for i in range(5):
        for j in range(5):
            if i == 4 and j == 4:
                break
            x = j*4
            y = (5-i)*4
            # 由於 k-means 分群的結果是隨機的
            # 所以要將分群結果和色板的顏色進行比較
            axs[1, 0].add_patch(plt.Rectangle((x, y), 4, 4, color=np.array(center_rgb[i * 5 + j]) / 255))
            axs[1, 1].add_patch(plt.Rectangle((x, y), 4, 4, color=np.array(center_rgb_clean[i * 5 + j]) / 255))
            axs[1, 2].add_patch(plt.Rectangle((x, y), 4, 4, color=np.array(rgb_list[i * 5 + j]) / 255))
            # 在方框中心位置添加 RGB 值文字
            rgb_text = f"({round(center_rgb[i * 5 + j][0],1)},{round(center_rgb[i * 5 + j][1],1)},{round(center_rgb[i * 5 + j][2],1)})"
            axs[1, 0].text(x + 2, y + 2, rgb_text, color='black', fontsize=6, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

            rgb_clean_text = f"({round(center_rgb_clean[i * 5 + j][0],1)},{round(center_rgb_clean[i * 5 + j][1],1)},{round(center_rgb_clean[i * 5 + j][2],1)})"
            axs[1, 1].text(x + 2, y + 2, rgb_clean_text, color='black', fontsize=6, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

            rgb_text = f"({round(rgb_list[i * 5 + j][0],1)},{round(rgb_list[i * 5 + j][1],1)},{round(rgb_list[i * 5 + j][2],1)})"
            axs[1, 2].text(x + 2, y + 2, rgb_text, color='black', fontsize=6, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    axs[1, 0].axis('scaled')
    axs[1, 1].axis('scaled')
    axs[1, 2].axis('scaled')

    plt.tight_layout()
    plt.savefig('res/colorblock.png', dpi=300, bbox_inches='tight')

    # 將 Delta E 值和顏色對應的索引轉換為 NumPy 陣列
    delta_e_values = np.array(rgb_rgbc_cmp)
    delta_e_values_std = np.array(rgb_std_cmp)
    delta_e_values_stdc = np.array(rgbc_std_cmp)
    color_indices = np.arange(len(rgb_list) - 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.bar(color_indices, delta_e_values)
    # 設定圖表標籤和標題
    plt.xlabel('Color Index')
    plt.ylabel('Delta E')
    plt.title('Colors and Remove outlier Color')

    plt.subplot(132)
    plt.bar(color_indices, delta_e_values_std)
    # 設定圖表標籤和標題
    plt.xlabel('Color Index')
    plt.ylabel('Delta E')
    plt.title('Colors and Standard Color')

    plt.subplot(133)
    plt.bar(color_indices, delta_e_values_stdc)
    # 設定圖表標籤和標題
    plt.xlabel('Color Index')
    plt.ylabel('Delta E')
    plt.title('Remove outlier Colors and Standard Color')

    plt.savefig('res/k-means.png', dpi=300, bbox_inches='tight')

    #------------------每一點的rgb值與色差------------------
    mean_C /= 24
    mean_E /= 24
    # print("C:{} | E:{} ".format(mean_C, mean_E))
    #print(mean_C, mean_E)
    return mean_C, mean_E, img_rect_
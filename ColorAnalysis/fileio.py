import os
import pandas as pd
import matplotlib.pyplot as plt

def find_result_dir(idx = 0): #　預設在result，並將重複的資料夾用編號重新命名
    dir = "result" 
    result_dir = "result"
    
    # 儲存的根目錄
    if not os.path.exists(dir):  # 如果文件夹不存在，创建文件夹
        os.makedirs(dir)
    
    #　儲存的目錄
    if os.path.exists(os.path.join(dir, f"{result_dir}")):
        idx += 1
    while os.path.exists(os.path.join(dir, f"{result_dir}_{idx}")):
        idx += 1
    if idx == 0:
        os.makedirs(os.path.join(dir, f"{result_dir}"))
        return os.path.join(dir, f"{result_dir}")
    else:
        os.makedirs(os.path.join(dir, f"{result_dir}_{idx}"))
        return os.path.join(dir, f"{result_dir}_{idx}")

def save_image_file(file, result_dir = "result", idx = 0): #　預設在result，並將重複的圖片用編號重新命名
    if not os.path.exists(result_dir):  # 如果文件夹不存在，创建文件夹
        os.makedirs(result_dir)
    if os.path.exists(os.path.join(result_dir, f"{file}.png")):
        idx += 1
    while os.path.exists(os.path.join(result_dir, f"{file}_{idx}.png")):
        idx += 1
    if idx == 0:
        plt.savefig(os.path.join(result_dir, f"{file}.png"))
    else:
        plt.savefig(os.path.join(result_dir, f"{file}_{idx}.png"))

def save_text_file(delta_e, file, result_dir = "result", idx = 0): #　預設在result，並將重複的文字檔用編號重新命名
    # 輸出為 excel 
    data = [[(i, j), f"{delta_e[i][j]:.3f}"] for i in range(5) for j in range(5)]
    # 建立 DataFrame 物件
    df = pd.DataFrame(data, columns=["position", "delta_e"])
    
    if not os.path.exists(result_dir):  # 如果文件夹不存在，创建文件夹
        os.makedirs(result_dir)
    if os.path.exists(os.path.join(result_dir, f"{file}.xlsx")):
        idx += 1
    while os.path.exists(os.path.join(result_dir, f"{file}_{idx}.xlsx")):
        idx += 1
    if idx == 0:
        df.to_excel(f"{result_dir}/{file}.xlsx", index=False)
    else:
        df.to_excel(f"{result_dir}/{file}_{idx}.xlsx", index=False)

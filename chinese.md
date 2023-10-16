# Give-ocean-a-piece-of-your-mind
[🇹🇼繁體中文](https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/blob/main/chinese.md) | [🌏English](https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/blob/main/README.md) | [📝論文準備中]()|

<p align="center">
  <img src="https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/blob/main/image/9.png?raw=true" alt="frame9125" width="600">
  <br>
  <a href="https://app.codacy.com/gh/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/7d2de1a21412457b83366b5e822cdfac">
</p>

## 摘要🧋
本系統致力於海洋永續發展，結合水下影像的上色與還原、物件偵測等功能，以多種神經網路模型，協助人們更有效地深入研究和認識海洋。

## 論文📝
'Enhancing Underwater Images: Automatic Colorization using Deep Learning and Image Enhancement Techniques,' 
2023 IEEE International Conference on Marine Artificial Intelligence and Law (IEEE ICMAIL 2023).

## 特色
+ **多媒體支援**：能夠載入圖片、影片、網路視訊鏡頭。
+ **場景多樣性**：提供水下、陸上等場景的權重。
+ **海洋永續科技**：協助海洋的觀察與研究。

<img src="https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/blob/main/image/5.png?raw=true" alt="frame9125" width="300"><img src="https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/blob/main/image/2.png?raw=true" alt="frame9125" width="308">

## 主要功能
1. **水下影像還原**：WaterNet，使用該作者提供的資料集訓練的權重。
2. **自動上色**：neural-colorization，該作者提供的原始權重與我們訓練的權重。
3. **載入圖片、影片、網路視訊鏡頭**：支援多媒體使用。
4. **影像的滑塊比對**：提供更便利與直觀的比對方式。
5. **物件辨識**：YOLO v8，官方原始權重與我們訓練的魚群、色板的權重。
6. **色板擷取與分析**：透過手動擷取色板，以評估影像中的色板資料。

<img src="https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/blob/main/image/7.png?raw=true" alt="frame9125" width="300"><img src="https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/blob/main/image/8.png?raw=true" alt="frame9125" width="300">
<img src="https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/blob/main/image/6.png?raw=true" alt="frame9125" width="250">

## Installation

+ 若要在您的本地端執行此專案，請依照以下步驟進行：

1. 執行以下指令來複製此專案的 repository：
```bash
git clone https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind.git
```

2. 執行以下指令安裝所需的套件：
```bash
pip install -r requirements.txt
```
這個指令會安裝此專案所需的所有套件。

注意：此專案使用的 python 版本為 3.8.10 ，若您的 python 版本差距過大，可能會有套件無法正常使用的問題。

## 使用方法

+ 執行 main.py ：
```bash
python3 main.py
```

## 注意
+ 開發環境為 Ubuntu 20.04.6 LTS
使用 windows 的用戶需要使用 `python main.py` 執行，並將程式碼中的所有 `python3` 改為 `python`。

範例：
```
subprocess.call([
                "python", colorization_path,
                "-i", source_path,
                "-m", weights_path,
                "-o", output_path,
                "--gpu", "-1",
            ]) 
```

## 研究方法
使用[WaterNet](https://github.com/tnwei/waternet)水色還原模型與[neural-colorization](https://github.com/zeruniverse/neural-colorization)自動上色模型來進行水下色彩還原、黑白影像上色，再使用數值方法評估影像還原結果，並透過[YOLO v8](https://github.com/ultralytics/ultralytics)進行物件辨識。
1. WaterNet水色還原模型
WaterNet是一個基於卷積神經網絡的模型，可以減少水下光散射所帶來的影響。
2. neural-colorization自動上色模型
由於水下紅光損失嚴重，而自動上色模型，可以在灰階影像中產生出任何顏色，故將其應用至水下色彩還原。
3. 色彩分析
為了比對影像還原前後的結果，我們將色板照片放入拍攝場景中，來分析評估還原的成效。並使用k-means與CIE-2000來對色塊進行分析。
4. 物件辨識
透過YOLO v8物件偵測模型，可以對各物件、魚類及色板進行辨識。

## 其他
+ 潛在問題：
  - colormath 的版本問題：可能造成色彩分析無法正常進行。
    - [Windows 第一分頁的色板分析無法正確分析 #27](https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/issues/27)
  - NumPy 的版本問題：
    - [手動擷取時，遇到 NumPy 的運行問題#26](https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/issues/26)
  - 視窗大小的隱患：在1080p解析度以下的電腦，可能難以使用。
    - [視窗大小的隱患 #17](https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/issues/17)
  - windows使用上的問題：
    - [在 windows 下 waterNet 與 colorization 運行錯誤的問題 #5](https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/issues/6)
+ 過去版本
  - [waternet_fasterRCNN](https://github.com/NTOU-Arrays-Start-at-One/waternet_fasterRCNN.git)：色板自動擷取與分析
  - [Perspective-control-and-Color-testing](https://github.com/NTOU-Arrays-Start-at-One/Perspective-control-and-Color-testing.git)：色板自動校正與分析

## 參考程式碼
- [WaterNet](https://github.com/tnwei/waternet)
- [neural-colorization](https://github.com/zeruniverse/neural-colorization)
- [YOLO v8](https://github.com/ultralytics/ultralytics)

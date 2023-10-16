# Give Ocean A Piece Of Your Mind

<div align="center">
  <div>
    <a href="https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/blob/main/README_zh_TW.md">🇹🇼繁體中文</a> |
    <a href="https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/blob/main/README.md">🌏English</a> |
    <a href="https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/issues">❓issues</a> |
    <a href="#">📝Paper Coming Soon</a>
  </div>
  <br>
  <img src="https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/assets/75748924/53e3f54e-0de8-4751-ac07-02628458ac09" alt="badge" width="700">
  <br>
  <div>
    <a href="https://app.codacy.com/gh/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/7d2de1a21412457b83366b5e822cdfac">
  </a>
    <img alt="Using Python version" src="https://img.shields.io/badge/python-3.8.10-blue.svg">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind">
  </div>
</div>

## Overview🍩
This system is dedicated to the sustainable development of the ocean, combining underwater image colorization and restoration, object detection, and various neural network models to assist people in more effectively researching and understanding the ocean.
<div align="center">
  <img src="https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/blob/main/image/9.png?raw=true" alt="frame9125" width="600">
</div>

## Paper📝
Enhancing Underwater Images: Automatic Colorization using Deep Learning and Image Enhancement Techniques, 2023 IEEE International Conference on Marine Artificial Intelligence and Law (IEEE ICMAIL 2023).

## Advantages
+   **Multimedia Support**: Capable of loading images, videos, and web camera feeds.
+   **Scene Variety**: Provides weights for underwater and land scenes.
+   **Ocean Sustainability Technology**: Assists in the observation and research of the ocean.

<div align="center">
  <img src="https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/blob/main/image/5.png?raw=true" alt="frame9125" width="300"><img src="https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/blob/main/image/2.png?raw=true" alt="frame9125" width="308">
</div>

## Key Features

1.  **Underwater Image Restoration**: WaterNet, using weights trained on the provided dataset by the author.

2.  **Automatic Colorization**: neural-colorization, original weights provided by the author, and weights trained by us.

3.  **Load Images, Videos, and Web Camera Feeds**: Supports multimedia usage.

4.  **Image Slider Comparison**: Provides a more convenient and intuitive way to compare images.

5.  **Object Recognition**: YOLO v8, official original weights and weights trained for fish and colorblock recognition.

6.  **Colorblock Capture and Analysis**: Manually captures colorblocks to evaluate colorblock data in images.

<div align="center">
  <img src="https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/blob/main/image/7.png?raw=true" alt="frame9125" width="290"><img src="https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/blob/main/image/8.png?raw=true" alt="frame9125" width="290"><img src="https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/blob/main/image/6.png?raw=true" alt="frame9125" width="243">
</div>

## Installation

+   To run this project locally, follow these steps:

1.  Clone the repository of this project by executing the following command:
```bash
git clone https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind.git
```

2.  Install the required packages by running the following command:
```bash
pip install -r requirements.txt
```
This will install all the necessary packages for this project.

Note: This project uses python version 3.8.10. If your python version is too different, there may be problems with the packages.

## Usage
+   Run main.py:
```bash
python3 main.py
```

## Notes📔
+   Development environment: Ubuntu 20.04.6 LTS
+   Windows users need to execute `python main.py` and replace all instances of `python3` in the code with `python`.

Example:
```python
subprocess.call([
                "python", colorization_path,
                "-i", source_path,
                "-m", weights_path,
                "-o", output_path,
                "--gpu", "-1",
            ]) 
```

## Research Methods
We employ the [WaterNet](https://github.com/tnwei/waternet) water color restoration model and [neural-colorization](https://github.com/zeruniverse/neural-colorization) automatic colorization model for underwater color restoration and black-and-white image colorization. Subsequently, we employ numerical methods to evaluate the image restoration results. Furthermore, we utilize [YOLO v8](https://github.com/ultralytics/ultralytics) for object recognition.

1.  WaterNet Water Color Restoration Model
WaterNet is a model based on convolutional neural networks designed to mitigate the impact of underwater light scattering.

2.  Neural-Colorization Automatic Colorization Model
Due to severe red light loss underwater, the automatic colorization model has the capability to generate any color in grayscale images. As a result, we apply it to underwater color restoration.

3.  Color Analysis
In order to compare results before and after image restoration, we incorporate colorboard photos into the shooting scenes to analyze and assess the effectiveness of restoration. We employ k-means and CIE-2000 for color block analysis.

4.  Object Recognition
Through YOLO v8 object detection, we can identify various objects, fish species, and colorboards.

## Other
### Potential Issues
  - colormath Version Problem: May result in color analysis not functioning correctly.
    - [Issue #27: Colorboard Analysis on the First Page of Windows Cannot Be Analyzed Correctly](https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/issues/27)
  - NumPy Version Problem:
    - [Issue #26: NumPy Runtime Issue Encountered When Capturing Manually](https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/issues/26)
  - Window Size Concern: Computers with resolutions below 1080p may have usability issues.
    - [Issue #17: Window Size Concerns](https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/issues/17)
  - Windows Usage Problems:
    - [Issue #6: Errors Running waterNet and Colorization on Windows](https://github.com/NTOU-Arrays-Start-at-One/Give-ocean-a-piece-of-your-mind/issues/6)

### Past Versions
  - [waternet_fasterRCNN](https://github.com/NTOU-Arrays-Start-at-One/waternet_fasterRCNN.git): Automatic color board capture and analysis.
  - [Perspective-control-and-Color-testing](https://github.com/NTOU-Arrays-Start-at-One/Perspective-control-and-Color-testing.git): Automatic color board calibration and analysis.

## Code References
-   [WaterNet](https://github.com/tnwei/waternet)
-   [neural-colorization](https://github.com/zeruniverse/neural-colorization)
-   [YOLO v8](https://github.com/ultralytics/ultralytics)

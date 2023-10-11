FROM fadawar/docker-pyqt5

WORKDIR /app

# 安裝pip
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install --upgrade pip

# 將 requirements.txt 複製到容器中
COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y python3
RUN pip install -r requirements.txt

COPY . /app

# 定义容器启动命令
CMD ["python3", "main.py"]

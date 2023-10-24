FROM ubuntu:20.04

ENV ACCEPT_EULA=Y DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt-get install -y software-properties-common &&\
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update &&\
    apt-get install -y python3.8 &&\
    apt-get install -y python3-pip &&\
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 10 &&\
    update-alternatives --set python3 /usr/bin/python3.8 &&\
    apt-get install -y python3-pyqt5

RUN pip3 install torch torchvision

WORKDIR /app

COPY . /app
RUN pip3 install -r /app/requirements.txt
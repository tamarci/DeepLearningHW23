
#FROM python:3.10.11
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt-get install -y python3.10 python3-pip
RUN apt update
RUN apt-get install -y python3.10 python3-pip nvidia-container-toolkit jupyter

RUN apt-get install 

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY train_and_evaluate.py .
RUN mkdir models
RUN mkdir data
COPY naroozi_perms_100_patches_9_max.npy data/

# Run everything in interactive terminal


#FROM python:3.10.11
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt-get install -y python3.10 python3-pip
RUN apt update
RUN apt-get install -y python3.10 python3-pip nvidia-container-toolkit

RUN apt-get install 

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
COPY startup.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/startup.sh
# COPY HF_Datadicts/ ./HF_Datadicts/

# CMD ["python3", "./train.py"]
CMD ["/usr/local/bin/startup.sh"]

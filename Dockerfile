FROM tensorflow/tensorflow:latest-gpu


RUN apt-get update -y
RUN apt-get install -y python3-pip git

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

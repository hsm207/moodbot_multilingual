FROM rasa/rasa:2.8.16-full

USER root

RUN apt update && \
   apt install -y make

RUN pip install black \
   debugpy \
   requests \
   textblob==0.17.1
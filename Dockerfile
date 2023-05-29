FROM nvcr.io/nvidia/pytorch:23.04-py3
RUN apt-get -y update && apt-get -y upgrade && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt-get install -y --no-install-recommends \
    ffmpeg
RUN pip install labelme
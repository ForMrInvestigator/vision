FROM nvcr.io/nvidia/pytorch:23.08-py3
RUN apt -y update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt install -y --no-install-recommends \
    ffmpeg python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev \
    build-essential cmake libedit-dev libxml2-dev llvm-dev
RUN pip install labelme apache-tvm -i https://mirrors.aliyun.com/pypi/simple/
WORKDIR /app/vision
ADD . .

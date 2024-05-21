FROM nvcr.io/nvidia/pytorch:23.04-py3

ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

RUN git config --global user.name "handleandwheel"
RUN git config --global user.email "liuyumingyouxiang@126.com"
RUN git config --global github.user "handleandwheel"
RUN git config --global github.token "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"
RUN git config --global http.proxy http[s]://handleandwheel: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@127.0.0.1:8888

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

# install ros
RUN apt update && \
    apt install lsb-release tzdata -y && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN pip install mujoco gymnasium mujoco-py
RUN apt update && apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev -y
RUN apt-get install python3-tk -y
RUN pip install torch_geometric
RUN ldconfig

WORKDIR /root

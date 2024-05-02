FROM nvcr.io/nvidia/cuda:11.7.0-runtime-ubuntu22.04
WORKDIR /workspace

# python, dependencies for mujoco-py, from https://github.com/openai/mujoco-py
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    build-essential \
    patchelf \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
# installing mujoco distr
RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

# installing poetry & env setup, mujoco_py compilation
COPY requirements/requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install "cython<3" ray h5py gymnasium[box2d]
RUN ["python", "-c", "import mujoco_py"]

RUN pip install pypatch
COPY lunarlander.patch lunarlander.patch
COPY lunarlander-seed.patch lunarlander-seed.patch 
RUN pypatch apply lunarlander.patch gymnasium
RUN pypatch apply lunarlander-seed.patch gymnasium

RUN mkdir checkpoints
RUN mkdir wandb
RUN mkdir jsrl-CORL
RUN pip install --upgrade pip
RUN pip install -i https://test.pypi.org/simple/ combination-lock==0.0.6


FROM python:3.12-bookworm

ARG DEBIAN_FRONTEND=noninteractive

# install lib required for pyaudio
RUN apt update && apt install -y portaudio19-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

# update pip
RUN pip install --no-cache-dir -U "pip>=24"
RUN pip install --no-cache-dir setuptools wheel

RUN mkdir /app
WORKDIR /app

# Установить стабильную PyTorch 2.4 (меньше по размеру, надежнее)
RUN pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Скопировать и отредактировать requirements.txt
COPY requirements.txt /app/
# Удалить строки с torch и torchaudio (они уже установлены)
RUN sed -i '/^torch/d; /^torchaudio/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib"
COPY whisper_live /app/whisper_live
COPY run_server.py /app
ENV TORCH_LOAD_WEIGHTS_ONLY=0
CMD ["python", "run_server.py", "--port", "8000", "--use_diarization", "-b", "faster_whisper"]

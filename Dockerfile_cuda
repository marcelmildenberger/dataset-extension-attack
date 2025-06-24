FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && \
    apt install -y libfreetype6-dev g++ intel-mkl nano && \
    apt clean && rm -rf /var/lib/apt/lists/*


# Copy Code
COPY ./ /usr/app/
WORKDIR /usr/app

RUN pip install --upgrade pip && pip install -r requirements.txt

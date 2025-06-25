FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirror.ubuntu.com/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirror.ubuntu.com/ubuntu|g' /etc/apt/sources.list && \
    apt update -o Acquire::Retries=5 && \
    apt install -y \
        libfreetype6-dev \
        g++ \
        nano


# Copy Code
COPY ./ /usr/app/
WORKDIR /usr/app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
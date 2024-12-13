# Stage 1: Base image with common dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Set bucket credentials
ARG BUCKET_ENDPOINT_URL
ARG BUCKET_ACCESS_KEY_ID
ARG BUCKET_SECRET_ACCESS_KEY

ENV BUCKET_ENDPOINT_URL=${BUCKET_ENDPOINT_URL}
ENV BUCKET_ACCESS_KEY_ID=${BUCKET_ACCESS_KEY_ID}
ENV BUCKET_SECRET_ACCESS_KEY=${BUCKET_SECRET_ACCESS_KEY}

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install comfy-cli
RUN pip install comfy-cli

# Install PyTorch separately
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install xformers
RUN pip install --no-cache-dir xformers

# Install ComfyUI with retry mechanism
RUN for i in 1 2 3; do \
    echo "Attempt $i: Installing ComfyUI..." && \
    /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 11.8 --nvidia --version 0.2.7 && \
    cd /comfyui/custom_nodes && \
    for j in 1 2 3; do \
        echo "Attempt $j: Installing ComfyUI-Manager..." && \
        git clone https://github.com/ltdrdata/ComfyUI-Manager && break || \
        echo "Git clone failed, cleaning up and retrying in 15 seconds..." && \
        rm -rf ComfyUI-Manager && \
        sleep 15; \
    done && \
    break || \
    echo "ComfyUI installation failed, retrying in 15 seconds..." && \
    rm -rf /comfyui/* && \
    sleep 15; \
done

# Verify installation
RUN if [ ! -d "/comfyui" ] || [ ! -d "/comfyui/custom_nodes/ComfyUI-Manager" ]; then \
    echo "Error: ComfyUI or ComfyUI-Manager installation failed!" && exit 1; \
fi

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install runpod
RUN pip install runpod requests

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add scripts
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py flux_de_distilled_input.json ./
RUN chmod +x /start.sh /restore_snapshot.sh

# Convert Windows line endings to Unix
RUN sed -i 's/\r$//' /start.sh /restore_snapshot.sh

# Optionally copy the snapshot file
ADD *snapshot*.json /

# Restore the snapshot to install custom nodes
RUN /restore_snapshot.sh

# Start container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base as downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG MODEL_TYPE

# Add error checking and verbose output for wget
RUN apt-get update && apt-get install -y curl

# Create necessary directories with proper permissions
WORKDIR /comfyui
RUN mkdir -p models/checkpoints models/vae models/unet models/clip models/unet&& \
    chmod -R 755 models

# Create necessary directories
RUN mkdir -p models/checkpoints models/vae models/unet models/clip

# Download checkpoints/vae/LoRA to include in image based on model type
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
      wget -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      wget -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
    elif [ "$MODEL_TYPE" = "sd3" ]; then \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
    elif [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
      wget -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
      wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
    elif [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      echo "Downloading FLUX.1-dev de-distilled models..." && \
      wget --verbose --show-progress --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        -O models/unet/real_korean_de_distilled-step00002500.safetensors \
        https://huggingface.co/trueorfalse441/real_korean_de_destilled_2nd/resolve/main/real_korean_de_distilled-step00002500.safetensors && \
      wget --verbose --show-progress --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        -O models/clip/clip_l.safetensors \
        https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget --verbose --show-progress --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        -O models/clip/t5xxl_fp16.safetensors \
        https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors && \
      wget --verbose --show-progress --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        -O models/vae/ae.safetensors \
        https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
    fi

# Stage 3: Final image
FROM base as final

# Copy models with explicit paths and verification
COPY --from=downloader /comfyui/models/ /comfyui/models/
RUN ls -la /comfyui/models/*

# Add a verification step
RUN if [ ! -f /comfyui/models/unet/real_korean_de_distilled-step00002500.safetensors ] || \
    [ ! -f /comfyui/models/clip/clip_l.safetensors ] || \
    [ ! -f /comfyui/models/clip/t5xxl_fp16.safetensors ] || \
    [ ! -f /comfyui/models/vae/ae.safetensors ]; then \
      echo "Error: Required models are missing!" && exit 1; \
    fi

# Start container
CMD ["/start.sh"]
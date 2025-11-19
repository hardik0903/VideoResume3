# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# 1. Install system dependencies required for OpenCV and Audio
# libgl1-mesa-glx: Required for OpenCV
# libglib2.0-0: Required for OpenCV
# libsndfile1: Required for librosa/soundfile
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy the application code
COPY . .

# 4. Create a non-root user for security (Hugging Face Spaces standard)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# 5. Expose port 7860 (Hugging Face default)
EXPOSE 7860

# 6. Command to run the app
# Note: Workers set to 2 for concurrency
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
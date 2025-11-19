# Use full Python image (not slim) to prevent segmentation faults with OpenCV/MediaPipe
FROM python:3.10

# Set working directory
WORKDIR /app

# 1. Install system dependencies
# We need these for OpenCV and Audio processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy application code
COPY . .

# 4. Create a user (Hugging Face requirement) and set permissions
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
# Ensure /tmp is writable for temp files
RUN chmod 777 /tmp

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 5. Expose the port
EXPOSE 7860

# 6. Run command
# Reduced workers to 1 to prevent memory overload on free tier
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
# Use a stable base image (Debian Bookworm)
FROM python:3.10-slim-bookworm

# Set working directory
WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ghostscript \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port
EXPOSE 5000

# Start app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]

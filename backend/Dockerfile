FROM python:3.10-slim

# Set the working directory directly
WORKDIR /app

# Install explicit system dependencies required by OCR (easyocr/camelot) and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Optimize layer caching: Install requirements before copying entire codebase
COPY requirements.txt .

# Install heavy ML stack (--no-cache-dir reduces output image size dramatically)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend codebase
COPY . .

# Expose standard FastAPI port
EXPOSE 8000

# Strict Uvicorn launch mapping
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

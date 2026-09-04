FROM python:3.10-slim

# Set the working directory directly
WORKDIR /app

# System dependencies:
#   build-essential  – compiling any sdist-only wheels
#   libgl1, libglib2.0-0 – OpenCV (imported by camelot / ultralytics / figure splitter)
#   poppler-utils    – PDF rasterisation helpers
#   ghostscript      – REQUIRED by camelot flavor='lattice' table extraction.
#                      Without it, table parsing silently returns nothing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# Optimize layer caching: Install requirements before copying entire codebase
COPY requirements.txt .

# Package index. Default is public PyPI. On an Alibaba Cloud ECS instance the
# build is far faster and far less likely to time out on the ~1.5 GB of ML
# wheels if you build with:
#   docker compose build --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG PIP_INDEX_URL=https://pypi.org/simple

# Install ML stack (--no-cache-dir keeps the image ~2 GB smaller).
# Generous retries/timeout: the TensorFlow + Torch wheels are large and stall
# on slow links, which otherwise fails the whole build.
RUN pip install --no-cache-dir --retries 5 --timeout 180 --prefer-binary \
    --index-url ${PIP_INDEX_URL} \
    -r requirements.txt

# Copy the rest of the backend codebase (includes models/*.pt weights)
COPY . .

# Expose standard FastAPI port
EXPOSE 8000

# Strict Uvicorn launch mapping
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

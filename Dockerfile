FROM python:3.11-slim

# System deps some scientific packages need to build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so Docker caches this layer unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual project
COPY . .

# Jupyter for exploration; container defaults to a shell for scripts
EXPOSE 8888

CMD ["bash"]

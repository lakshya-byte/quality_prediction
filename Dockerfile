# Use official Python image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system deps (optional but good for scientific libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Flask port
EXPOSE 8000

# Start the app
CMD ["python", "main.py"]

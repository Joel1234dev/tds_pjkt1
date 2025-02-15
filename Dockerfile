# Use Python 3.9 as base image
FROM python:3.9-slim

# Install Node.js and npm
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=development
# AIPROXY_TOKEN will be passed at runtime

# Expose port 8000 instead of 5000
EXPOSE 8000

# Run the application on port 8000
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"] 
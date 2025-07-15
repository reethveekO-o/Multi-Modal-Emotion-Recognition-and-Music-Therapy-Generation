# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (except those in .gitignore)
COPY . .

# (Optional) Set environment variable for unbuffered Python output
ENV PYTHONUNBUFFERED=1

# Set the default command to run your main script
CMD ["python", "main.py"]

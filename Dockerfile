FROM python:3.11-slim

WORKDIR /app

# Install git (add this line before pip install)
RUN apt-get update && \
    apt-get install -y build-essential gcc gfortran && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt


COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
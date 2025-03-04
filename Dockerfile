FROM python:3.11.11-bookworm

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install -r requirements.txt
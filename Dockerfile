FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY . .

RUN python -m ensurepip --upgrade
RUN python -m pip install --no-cache-dir numpy==1.26.4
RUN python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["gunicorn", "app_nodb:app", "--bind", "0.0.0.0:8080"]

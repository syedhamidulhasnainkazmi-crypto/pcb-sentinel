FROM python:3.10-slim 
WORKDIR /app 
COPY . . 
RUN pip install --no-cache-dir numpy==1.26.4 
RUN pip install --no-cache-dir -r requirements.txt 
EXPOSE 8080 
CMD ["gunicorn", "app_nodb:app", "--bind", "0.0.0.0:8080"] 

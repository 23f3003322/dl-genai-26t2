FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway automatically passes the $PORT variable, which Gradio reads above
EXPOSE 7860

CMD ["python", "app.py"]

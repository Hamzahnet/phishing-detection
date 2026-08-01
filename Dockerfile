FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

RUN python train_model.py

CMD ["python", "predict.py", "http://example.com"]
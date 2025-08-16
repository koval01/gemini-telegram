FROM python:3.12.6-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements/ requirements/

RUN pip install --upgrade pip \
        && pip install -r requirements/production.txt \
        && rm -rf requirements

COPY . .

CMD ["python", "./bot.py"]

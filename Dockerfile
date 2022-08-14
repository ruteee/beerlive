FROM python:3.9

WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

COPY . /app

CMD [ "python", "./app.py" ]

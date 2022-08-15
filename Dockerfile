FROM python:3.8

WORKDIR /app

COPY requirements.txt /app/requirements.txt


COPY linux_requirements.sh /app/linux_requirements.sh

RUN pip install --no-cache-dir -r /app/requirements.txt

RUN chmod +x /app/linux_requirements.sh
RUN  /app/linux_requirements.sh

EXPOSE 8080

COPY . /app

CMD [ "python", "./app.py" ]

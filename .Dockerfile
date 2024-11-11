FROM --platform=linux/amd64 python:3.11.4-bookworm as build

WORKDIR /root/app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY ./app /root/app

CMD tail -f /dev/null
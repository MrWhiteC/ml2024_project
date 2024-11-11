FROM --platform=linux/amd64 python:3.11.4-bookworm as build

WORKDIR /root/web

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY ./web /root/web

CMD tail -f /dev/null
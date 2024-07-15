FROM python:3.10

RUN apt update

RUN apt install python3 -y

RUN apt-get install python3-pip -y

WORKDIR /wellnest-ecg-ai

COPY . /wellnest-ecg-ai

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python3", "app.py"]


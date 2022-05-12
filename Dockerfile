FROM python:3.10-slim

WORKDIR /app

RUN apt-get update -y
RUN apt-get install -y gifsicle

COPY requirements.txt /app/

RUN pip install -r requirements.txt

COPY images/ images/

COPY main.py main.py
COPY src/ src/

ENTRYPOINT ["python", "main.py"]
CMD ["10"]

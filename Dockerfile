FROM python:3.9 

COPY . /app

WORKDIR /app

# install dependencies with pip
RUN pip install -r requirements.txt

CMD ["python", "main.py"]

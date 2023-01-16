FROM python:3.9 

COPY . /app

WORKDIR /app

# install poetry 
RUN pip install poetry

# dump poetry dependencies
RUN poetry export -f requirements.txt --output requirements.txt

# install dependencies with pip
RUN pip install -r requirements.txt

CMD ["python", "main.py"]
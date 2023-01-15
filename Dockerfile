FROM python:3.9

COPY . /app

WORKDIR /app

# install poetry 
RUN pip install poetry

# install dependencies with poetry

RUN poetry install

CMD ["python", "main.py"]
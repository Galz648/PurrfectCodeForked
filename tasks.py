from celery import Celery, chain

app = Celery('tasks', broker='pyamqp://guest@localhost//', backend='redis://localhost')

@app.task
def add(x, y) -> int:
    return x + y

@app.task
def multiply(result: int, factor):
    return result * factor

# Define the pipeline using the chain method
pipeline = chain(add.s(2, 2), multiply.s(10))

# Execute the pipeline and get the final result

if __name__ == "__main__":
    result = pipeline()
    print(result.get())
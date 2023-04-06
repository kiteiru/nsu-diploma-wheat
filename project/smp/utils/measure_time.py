import time

def measure_time(function):
    def wrapper(*args, **kwargs):
        start = time.time()
        function(*args, **kwargs)
        end = time.time()

        print(f"Time elapsed: {round((end - start) / 60, 4)} minutes")

    return wrapper
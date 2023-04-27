from itertools import islice


def take(iterable, n):
    return list(islice(iterable, n))

def unchain(iterable, n):
    iterable = iter(iterable)

    while True:
        result = take(iterable, n)
        if result:
            yield result
        else:
            break

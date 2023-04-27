import numpy as np

def batch_ids_fullsize_generator(size, batch_size, shuffle=False):
    """
        NOTE:
            the last batch size might be not equal to batch_size
    """
    ids = np.arange(size)

    if shuffle:
        np.random.shuffle(ids)

    poses = np.arange(batch_size, size, batch_size)
    return np.split(ids, poses)

def batch_ids_fixedsize_generator(size, batch_size, n_count=1):
    assert n_count > 0

    for _ in range(n_count):
        yield np.random.choice(size, size=batch_size)

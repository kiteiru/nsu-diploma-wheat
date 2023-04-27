import numpy as np

from itertools import product, chain

def tile_random_selector(tile_shape, shapes, n_count=1):
    n_objects = np.arange(len(shapes))

    for _ in np.arange(n_count):
        idx = np.random.choice(n_objects)

        shape = shapes[idx]

        selector = ()

        for tile_size, size in zip(tile_shape, shape):
            if tile_size is None:
                selector = (*selector, slice(0, None))
            else:
                assert tile_size <= size

                if tile_size == size:
                    coord = 0
                else:
                    coord = np.random.choice(size - tile_size)

                selector = (*selector, slice(coord, coord + tile_size))

        yield idx, selector

def tile_sequential_selector(tile_shape, shapes, steps):

    for idx, shape in enumerate(shapes):
        ranges = ()

        for step, tile_size, size in zip(steps, tile_shape, shape):
            assert size >= tile_size

            range_ = chain(np.arange(0, size - tile_size, step), (size - tile_size ,))
            ranges = (*ranges, range_)

        for point in product(*ranges):
            selector = ()

            for coord, tile_size in zip(point, tile_shape):
                selector = (*selector, slice(coord, coord+tile_size))

            yield idx, selector

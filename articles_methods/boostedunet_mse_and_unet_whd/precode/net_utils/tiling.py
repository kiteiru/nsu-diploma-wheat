import cv2
import numpy as np

from itertools import product, chain

def __get_paddings(initial_size, required_size):
    if initial_size < required_size:
        pad_left = int((required_size - initial_size) / 2.0)
        pad_right = required_size - initial_size - pad_left
    else:
        pad_left = 0
        pad_right = 0

    return pad_left, pad_right

def pad_tiles( image, height, width,
               *,
               border_mode=cv2.BORDER_REFLECT_101,
               value=None,
               mask=None,
               mask_value=None ):
    iheight = image.shape[0]
    iwidth = image.shape[1]

    assert iheight <= height
    assert iwidth <= width

    h_pad_top, h_pad_bottom = __get_paddings(iheight, height)
    w_pad_left, w_pad_right = __get_paddings(iwidth, width)

    paddings = { 'top': h_pad_top,
                 'bottom': h_pad_bottom,
                 'left': w_pad_left,
                 'right': w_pad_right }

    pad_image = cv2.copyMakeBorder(image, **paddings, borderType=border_mode, value=value)

    pad_loss_mask = np.zeros((height, width))

    xslice = slice(paddings['top'], -paddings['bottom'])
    yslice = slice(paddings['left'], -paddings['right'])

    pad_loss_mask[xslice, yslice] = 1

    pad_output = (pad_image, pad_loss_mask)

    if mask is not None:
        pad_mask = cv2.copyMakeBorder(mask, **paddings, borderType=border_mode, value=mask_value)
        pad_output = (*pad_output, pad_mask)

    return pad_output, paddings

def __shape_assertions(image, mask, loss_mask, htile_size, wtile_size):
    assert image.shape[1:] == loss_mask.shape

    if mask is not None:
        assert mask.shape == loss_mask.shape

    assert image.shape[1] % htile_size == 0
    assert image.shape[2] % wtile_size == 0

def get_tiles( image,
               loss_mask,
               htile_size,
               wtile_size,
               *,
               mask=None):
    __shape_assertions(image, mask, loss_mask, htile_size, wtile_size)

    sizes = image.shape[1:]
    tile_sizes = (htile_size, wtile_size)

    for selector in tile_iterator(sizes, tile_sizes, tile_sizes):
        point = ()

        for slice_, tile_size in zip(selector, tile_sizes):
            point = (*point, slice_.start // tile_size)

        tile_output = (image[(slice(0, None), *selector)], loss_mask[selector])

        if mask is not None:
            tile_output = (*tile_output, mask[selector])

        yield ( tile_output, point )

def place_tile(datum, tile, tile_mask, hidx, widx, htile_size, wtile_size):
    hsid = hidx*htile_size
    heid = (hidx+1)*wtile_size

    wsid = widx*htile_size
    weid = (widx+1)*wtile_size

    datum[hsid:heid, wsid:weid] = tile * tile_mask

def __get_overlapped_tile_spans(idx, tile_size, img_size):
    tile_hsize = tile_size // 2

    eid = (idx + 1) * tile_hsize
    sid = eid - tile_size

    pads = ( )

    if eid > img_size:
        pads = (tile_hsize, *pads)
        eid = img_size
    else:
        pads = (0, *pads)

    if sid < 0:
        pads = (tile_hsize, *pads)
        sid = 0
    else:
        pads = (0, *pads)

    return slice(sid, eid), pads

def get_overlapped_tiles( image,
                          loss_mask,
                          htile_size,
                          wtile_size,
                          *,
                          mask=None):
    __shape_assertions(image, mask, loss_mask, htile_size, wtile_size)

    assert htile_size % 2 == 0
    assert wtile_size % 2 == 0

    htile_hsize = htile_size // 2
    wtile_hsize = wtile_size // 2

    hcount = image.shape[1] // htile_hsize
    wcount = image.shape[2] // wtile_hsize

    for hidx in np.arange(hcount+1):
        for widx in np.arange(wcount+1):
            xslice, xpads = __get_overlapped_tile_spans( hidx,
                                                         htile_size,
                                                         htile_hsize*hcount )

            yslice, ypads = __get_overlapped_tile_spans( widx,
                                                         wtile_size,
                                                         wtile_hsize*wcount )

            tile_output = (
                np.pad( image[:, xslice, yslice],
                            pad_width=((0, 0), xpads, ypads),
                            mode='symmetric' ),
                np.pad( loss_mask[xslice, yslice],
                            pad_width=(xpads, ypads),
                            mode='constant' ),
            )

            if mask is not None:
                tile_output = (
                    *tile_output,
                    np.pad( mask[xslice, yslice],
                            pad_width=(xpads, ypads),
                            mode='constant' )
                )

            yield ( tile_output, (hidx, widx) )

def __get_overlapped_tile_place(idx, tile_size, img_size):
    eid = (idx+1) * (tile_size // 2)
    sid = eid - tile_size

    eid -= (tile_size // 4)
    sid += (tile_size // 4)

    pads = ( )

    if sid < 0:
        pads = (*pads, (tile_size // 2) )
        sid = 0
    else:
        pads = (*pads, (tile_size // 4))

    if eid > img_size:
        pads = (*pads, (tile_size // 2) )
        eid = img_size
    else:
        pads = (*pads, (tile_size // 2) + (tile_size // 4))

    return slice(sid, eid), slice(*pads)

def place_overlapped_tile(datum, tile, tile_mask, hidx, widx, htile_size, wtile_size):
    assert htile_size % 4 == 0
    assert wtile_size % 4 == 0

    xslice, xtile_slice = __get_overlapped_tile_place(hidx, htile_size, datum.shape[0])
    yslice, ytile_slice = __get_overlapped_tile_place(widx, wtile_size, datum.shape[1])

    datum[xslice, yslice] = (tile * tile_mask)[xtile_slice, ytile_slice]

def tile_iterator(shape, tile_shape, steps):
    assert len(tile_shape) == len(steps) == 2

    ranges = ()

    for step, tile_size, size in zip(steps, tile_shape, shape):
        assert size >= tile_size

        range_ = chain(np.arange(0, size - tile_size, step), (size - tile_size ,))
        ranges = (*ranges, range_)

    for point in product(*ranges):
        selector = ()

        for coord, tile_size in zip(point, tile_shape):
            selector = (*selector, slice(coord, coord+tile_size))

        yield selector

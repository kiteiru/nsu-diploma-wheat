import torch
import types
import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as sm

def __classinit(cls):
    return cls._init__class()

def __is_generator_empty(generator):
    try:
        next(generator)
        return False
    except StopIteration:
        return True

def convert_inplace(net, convertor):
    stack = [net]

    while stack:
        node = stack[-1]

        stack.pop()

        for name, child in node.named_children():
            if not __is_generator_empty(child.children()):
                stack.append(child)

            setattr(node, name, convertor(child))

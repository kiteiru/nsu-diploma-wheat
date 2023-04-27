from nn_tools.metrics.misc import __score_generator
from nn_tools.metrics.conf import SegmentationConfusions

jaccard_score = __score_generator(lambda TP, FP, FN: TP / (TP + FP + FN), SegmentationConfusions)
dice_score = __score_generator(lambda TP, FP, FN: 2*TP / (2*TP + FP + FN), SegmentationConfusions)

precision_score = __score_generator(lambda TP, FP, FN: TP / (TP + FP), SegmentationConfusions)
recall_score = __score_generator(lambda TP, FP, FN: TP / (TP + FN), SegmentationConfusions)
f1_score = __score_generator(lambda TP, FP, FN,: 2*TP / (2*TP + FP + FN), SegmentationConfusions)

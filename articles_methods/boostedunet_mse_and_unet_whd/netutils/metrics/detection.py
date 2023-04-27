from netutils.metrics.misc import __score_generator
from netutils.metrics.confusions import PoundDetectionConfusions

precision_score = __score_generator(lambda TP, FP, FN: TP / (TP + FP), PoundDetectionConfusions)
recall_score = __score_generator(lambda TP, FP, FN: TP / (TP + FN), PoundDetectionConfusions)
f1_score = __score_generator(lambda TP, FP, FN,: 2*TP / (2*TP + FP + FN), PoundDetectionConfusions)

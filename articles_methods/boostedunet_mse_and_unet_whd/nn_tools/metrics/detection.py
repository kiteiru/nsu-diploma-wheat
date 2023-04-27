from nn_tools.metrics.misc import __score_generator
from nn_tools.metrics.conf import SeedDetectionConfusions

precision_score = __score_generator(lambda TP, FP, FN: TP / (TP + FP), SeedDetectionConfusions)
recall_score = __score_generator(lambda TP, FP, FN: TP / (TP + FN), SeedDetectionConfusions)
f1_score = __score_generator(lambda TP, FP, FN,: 2*TP / (2*TP + FP + FN), SeedDetectionConfusions)

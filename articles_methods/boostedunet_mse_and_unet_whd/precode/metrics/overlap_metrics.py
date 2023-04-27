from precode.metrics.misc import __score_generator

jaccard_score = __score_generator(lambda TP, FP, FN: TP / (TP + FP + FN))
dice_score = __score_generator(lambda TP, FP, FN: 2*TP / (2*TP + FP + FN))

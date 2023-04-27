import numpy as np

from collections import defaultdict


def confusions(true, pred):
    confusions = defaultdict(dict)

    for label in np.unique(true):
        selected_true = (true == label)
        selected_pred = (pred == label)

        confusions[label]['TP'] = np.sum((selected_true == selected_pred) & selected_true)
        confusions[label]['FP'] = np.sum((selected_true != selected_pred) & np.logical_not(selected_true))
        confusions[label]['FN'] = np.sum((selected_true != selected_pred) & np.logical_not(selected_pred))

    return confusions

def __global(conf):
    gTP, gFP, gFN = 0, 0, 0

    for key in conf:
        gTP += conf[key]['TP']
        gFP += conf[key]['FP']
        gFN += conf[key]['FN']

    return gTP, gFP, gFN

def __score_generator(__score):
    def __body(true, pred, average='micro'):
        conf = confusions(true, pred)

        if average == 'micro':
            gTP, gFP, gFN = __global(conf)

            return (__score(gTP, gFP, gFN), )
        elif average == 'macro':
            scores = list()

            for key in conf:
                score = __score( conf[key]['TP'],
                                 conf[key]['FP'],
                                 conf[key]['FN'] )
                scores.append(score)

            return (sum(scores) / len(scores), )
        elif average == 'none':
            skeys = sorted(conf.keys())
            return tuple( __score( conf[key]['TP'], conf[key]['FP'], conf[key]['FN'] ) for key in skeys )
        else:
            raise ValueError()

    return __body

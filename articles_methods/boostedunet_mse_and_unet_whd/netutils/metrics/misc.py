def __global(conf):
    gTP, gFP, gFN = 0, 0, 0

    for key in conf:
        gTP += conf[key]['TP']
        gFP += conf[key]['FP']
        gFN += conf[key]['FN']

    return gTP, gFP, gFN

def __score_zero_division_catcher(score):
    def callee(*args, **kwargs):
        try:
            return score(*args, **kwargs)
        except ZeroDivisionError:
            return 0

    return callee

def __score_generator(_score, confusions):
    def __body(true, pred, average, **kwargs):
        assert average in confusions.allowed_average_types(), 'Not allowed average type'

        conf = confusions.collect(true, pred, **kwargs)

        __score = __score_zero_division_catcher(_score)

        if average == 'binary':
            return (__score( conf[1]['TP'],
                             conf[1]['FP'],
                             conf[1]['FN']), )
        elif average == 'micro':
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

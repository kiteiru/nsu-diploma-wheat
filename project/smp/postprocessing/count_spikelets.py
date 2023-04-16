import numpy as np

def RMSE(t, d):
    n = len(t)
    summa = 0
    for i in range(n):
        summa += (t[i] - d[i]) ** 2
    result = np.sqrt(summa / n)
    return result

def rRMSE(t, d):
    n = len(t)
    summa = 0
    for i in range(n):
        summa += ((t[i] - d[i]) / t[i]) ** 2
    result = np.sqrt(summa / n)
    return result

def R_squared(t, d):
    n = len(t)
    t_mean = np.mean(t)
    summa1 = 0
    summa2 = 0
    for i in range(n):
        summa1 += (t[i] - d[i]) ** 2
        summa2 +=  (t[i] - t_mean) ** 2
    result = 1 - summa1 / summa2
    return result

def MAE(t, d):
    n = len(t)
    summa = 0
    for i in range(n):
        summa += abs(t[i] - d[i])
    result = summa / n
    return result
    
def MAPE(t, d):
    n = len(t)
    summa = 0
    for i in range(n):
        summa += abs((t[i] - d[i]) / t[i])
    result = summa * 100 / n
    return result


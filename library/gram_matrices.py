import numpy as np
from kgof.kernel import KGauss

def kgof_make_Ks(X, Px, bandwidth = 1):
    
    n, d = X.shape
    ndV = Px.shape[0]

    kG = KGauss(bandwidth)
    K = kG.eval(X, X)

    K_y = []
    for i in range(d):
        K_y.append(kG.gradY_X(X, X, i))

    
    Kyt = kG.eval(X, Px)
    Kxtyt = kG.eval(Px, Px)

    Kyt_x = []
    for i in range(d):
        Kyt_x.append(kG.gradX_Y(X, Px, i))

    return K, K_y, Kyt, Kyt_x, Kxtyt

def kgof_make_Ks_all(X, Px, bandwidth = 1):
    
    n, d = X.shape
    ndV = Px.shape[0]

    kG = KGauss(bandwidth)
    K = kG.eval(X, X)
    K_xy = kG.gradXY_sum(X, X)

    K_y = []
    for i in range(d):
        K_y.append(kG.gradY_X(X, X, i))
    
    Kyt = kG.eval(X, Px)
    Kxtyt = kG.eval(Px, Px)

    Kyt_x = []
    for i in range(d):
        Kyt_x.append(kG.gradX_Y(X, Px, i))

    return K, K_y, K_xy, Kyt, Kyt_x, Kxtyt


def kgof_make_Ks_X(X, Px, X0, bandwidth):
    
    n, d = X.shape
    kG = KGauss(bandwidth)
    k1 = kG.eval(X, X0)
    kp = kG.eval(Px, X0)

    k2 = []
    for i in range(d):
        k2.append(kG.gradX_Y(X, X0, i))

    return k1, k2, kp

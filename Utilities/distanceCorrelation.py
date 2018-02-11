import numpy as np


def dist(x, y):
    # 1d only
    return np.abs(x[:, None] - y)


def d_n(x):
    d = dist(x, x)
    dn = d - d.mean(0) - d.mean(1)[:, None] + d.mean()
    return dn


def dcov_all(x, y):
    x = np.array(x)
    y = np.array(y)
    dnx = d_n(x)
    dny = d_n(y)

    denom = np.product(dnx.shape)
    dc = (dnx * dny).sum() / denom
    dvx = (dnx ** 2).sum() / denom
    dvy = (dny ** 2).sum() / denom
    dr = dc / (np.sqrt(dvx) * np.sqrt(dvy))
    return dc, dr, dvx, dvy
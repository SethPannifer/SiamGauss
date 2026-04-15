import numpy as np


def MPMOP_Value(probID, x, t):
    x = np.asarray(x)
    N, n = x.shape

    if probID == 'MPMOP1':
        a = 5 * np.cos(0.5 * np.pi * t)
        tmp = 1 / (1 + np.exp(a * (x[:, 0] - 2.5)))
        g = 1 + np.sum((x[:, 1:] - tmp[:, None]) ** 2, axis=1)
        f1 = g * (1 + t) / x[:, 0]
        f2 = g * x[:, 0] / (1 + t)
        return np.column_stack((f1, f2))

    elif probID == 'MPMOP2':
        G = np.sin(0.5 * np.pi * t)
        a = 2.25 + 2 * np.cos(2 * np.pi * t)
        tmp = G * np.sin(4 * np.pi * x[:, 0]) / (1 + abs(G))
        g = 1 + np.sum((x[:, 1:] - tmp[:, None]) ** 2, axis=1)
        f1 = g * (x[:, 0] + 0.1 * np.sin(3 * np.pi * x[:, 0]))
        f2 = g * (1 - x[:, 0] + 0.1 * np.sin(3 * np.pi * x[:, 0])) ** a
        return np.column_stack((f1, f2))

    elif probID == 'MPMOP3':
        Nval = 1 + np.floor(10 * abs(np.sin(0.5 * np.pi * t))).astype(int)
        g = np.ones(N)

        for i in range(1, n):
            tmp = x[:, i] - np.cos(4 * t + x[:, 0] + x[:, i - 1])
            g += tmp ** 2

        term = np.maximum(0, (0.1 + 0.5 / Nval) * np.sin(2 * Nval * np.pi * x[:, 0]))
        f1 = g * (x[:, 0] + term)
        f2 = g * (1 - x[:, 0] + term)
        return np.column_stack((f1, f2))

    elif probID == 'MPMOP4':
        G = np.sin(0.5 * np.pi * t)
        H = 2.25 + 2 * np.cos(0.5 * np.pi * t)
        tmp = np.sin(2 * np.pi * (x[:, 0] + x[:, 1])) / (1 + abs(G))
        g = 1 + np.sum((x[:, 2:] - tmp[:, None]) ** 2, axis=1)

        f1 = g * (np.sin(0.5 * np.pi * x[:, 0]) ** H)
        f2 = g * (np.sin(0.5 * np.pi * x[:, 1]) ** H) * (np.cos(0.5 * np.pi * x[:, 0]) ** H)
        f3 = g * (np.cos(0.5 * np.pi * x[:, 1]) ** H) * (np.cos(0.5 * np.pi * x[:, 0]) ** H)

        return np.column_stack((f1, f2, f3))

    elif probID == 'MPMOP5':
        G = abs(np.sin(0.5 * np.pi * t))
        g = 1 + np.sum((x[:, 2:] - 0.5 * G * x[:, 0][:, None]) ** 2, axis=1)

        y = np.pi * G / 6 + (np.pi / 2 - np.pi * G / 3) * x[:, 0:2]

        f1 = g * np.sin(y[:, 0])
        f2 = g * np.sin(y[:, 1]) * np.cos(y[:, 0])
        f3 = g * np.cos(y[:, 1]) * np.cos(y[:, 0])

        return np.column_stack((f1, f2, f3))

    elif probID == 'MPMOP6':
        k = np.floor(10 * np.sin(np.pi * t))
        r = 1 - np.mod(k, 2)

        tmp1 = x[:, 2:] - np.sin(t * x[:, 0])[:, None]
        tmp2 = np.abs(np.sin(np.floor(k * (2 * x[:, 0:2] - r)) * np.pi / 2))

        g = 1 + np.sum(tmp1 ** 2, axis=1) + np.prod(tmp2, axis=1)

        f1 = g * np.cos(0.5 * np.pi * x[:, 1]) * np.cos(0.5 * np.pi * x[:, 0])
        f2 = g * np.sin(0.5 * np.pi * x[:, 1]) * np.cos(0.5 * np.pi * x[:, 0])
        f3 = g * np.sin(0.5 * np.pi * x[:, 0])

        return np.column_stack((f1, f2, f3))

    else:
        raise ValueError("No such test problem.")

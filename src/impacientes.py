# -*- coding: utf-8 -*-
from typing import List
import numpy as np
from random import expovariate


def simula_atendimento(
    n: int,
    lambd: int,
    mi: int,
    T: int,
) -> List[float]:
    """
    Simula uma versao mais simples do problema
    do atendimento, neste caso os clientes nao
    sao impacientes.

    Parameters
    ---

    n: int
        numero de linhas para atendimento.

    lambd: int
        taxa de atendimento.

    mi: float
        taxa do tempo de atendimento.

    T: int
        tempo total de observacao do
        experimento.

    Returns
    ---

    x: int
        numero de requisicoes aceitas.

    y: int
        numero de requisicoes rejeitadas.

    w: float
        proporcao de requisicoes rejeitadas.
    """
    tc = 0  # instante da chegada do ultimo cliente
    gt_disp = np.zeros(n)  # tempo de disponibilidade das linhas
    k = 0  # contador de clientes que entraram na fila
    ct_cheg = np.zeros(9999)  # instante em que o k-esimo cliente chega
    x, y, r, w = 0, 0, 0, 0  # aceitas e rejeitadas, comprimento atual da fila
    tm = 0  # tempo maximo de espera ate o momento

    z = expovariate(lambd)

    while tc + z <= T:
        tc += z
        k += 1
        ct_cheg[k] = tc

        while gt_disp[np.argmin(gt_disp)] <= tc and x < k:
            x += 1
            j = np.argmin(gt_disp)
            a = expovariate(mi)
            gt_disp[j] = max(gt_disp[j], ct_cheg[x]) + a
            tm = max(tm, (gt_disp[j] - ct_cheg[x]))

        r = max(0, (k - 1) - x)
        pr = r / (r + n)
        s = np.random.binomial(1, pr)

        if s == 1:
            k -= 1
            y += 1

        r = k - x
        w = y / (x + y + r)
        z = expovariate(lambd)

    return x, y, w, tm


def simula_atendimento_wrapper(
    n: int,
    lambd: int,
    mi: float,
    T: int,
    N: int,
) -> List[List[float]]:
    simulations = (N//100)+1 if N % 2 != 0 else N//100
    X = np.zeros(simulations)
    Y = np.zeros(simulations)
    W = np.zeros(simulations)
    TM = np.zeros(simulations)

    for idx, i in enumerate(range(0, N, 100)):
        xi, yi, wi, tmi = simula_atendimento(n, lambd, mi, T)
        X[idx] = xi
        Y[idx] = yi
        W[idx] = wi
        TM[idx] = tmi

    return X.mean(), Y.mean(), W.mean(), TM.mean()


if __name__ == "__main__":
    ux, uy, uw, utm = simula_atendimento_wrapper(5, 3, .5, 50, 20000)
    print(f"media aceita: {ux}")
    print(f"media rejeitada: {uy}")
    print(f"proporcao de rejeitadas: {uw}")
    print(f"tempo maximo (media): {utm}")

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
    ct_cheg = list()  # instante em que o k-esimo cliente chega
    x, y, r, w = 0, 0, 0, 0  # aceitas e rejeitadas, comprimento atual da fila
    tm = 0  # tempo maximo de espera ate o momento

    z = expovariate(lambd)

    while tc + z <= T:
        tc += z
        k += 1
        ct_cheg.append(tc)

        while gt_disp.min() <= tc and x < k:
            j = np.argmin(gt_disp)
            a = expovariate(mi)
            gt_disp[j] = max(gt_disp[j], ct_cheg[x]) + a
            tm = max(tm, (gt_disp[j] - ct_cheg[x]))
            x += 1

        r = max(0, (k - 1) - x)
        pr = r / (r + n)
        s = np.random.binomial(1, pr)

        if s == 1:
            ct_cheg.pop(-1)
            k -= 1
            y += 1

        r = k - x
        w = y / (x + y + r)
        z = expovariate(lambd)

    return x, y, w, tm


def intervalo_confianca(v: List, size: int) -> float:
    return 2 * 1.96 * std_error(v, size)


def std_error(v: List, size: int) -> float:
    v = np.array(v)
    return v.std() / np.sqrt(size)


def simula_atendimento_wrapper(
    n: int,
    lambd: int,
    mi: float,
    T: int,
) -> List[List[float]]:
    simulations = 100000
    steps = 100
    idx = 0
    idx_steps = 0
    interv_confianca = 1

    X = []
    Y = []
    W = []
    TM = []
    LS_w = []
    LI_w = []
    LS_tm = []
    LI_tm = []

    while interv_confianca >= 0.005:

        for i in range(steps):
            xi, yi, wi, tmi = simula_atendimento(n, lambd, mi, T)
            X.append(xi)
            Y.append(yi)
            W.append(wi)
            TM.append(tmi)

        interv_confianca = intervalo_confianca(W, len(W))
        print(f"{len(W)} | intervalo de confianca {intervalo_confianca(W, len(W))}", end='\r')
    
        #! create LS, LI vectors

        std_error_w = std_error(W, len(W))
        std_error_tm = std_error(TM, len(TM))

        LS_w.append(mean(W) + std_error_w)
        LI_w.append(mean(W) - std_error_w)
        LS_tm.append(mean(TM) + std_error_tm)
        LI_tm.append(mean(TM) - std_error_tm)

    simulacoes = [X, Y, W, TM]
    erros_padrao = [LS_w, LI_w, LS_tm, LI_tm]
    medias_moveis = [
        media_movel(X, len(X), steps),
        media_movel(Y, len(Y), steps),
        media_movel(W, len(W), steps),
        media_movel(TM, len(TM), steps)
    ]
    k_set = [steps*i for i in range(1, (len(W)//steps)+1)]

    return simulacoes, medias_moveis, erros_padrao, k_set


def mean(v):
    return np.array(v).mean()


def media_movel(X: List, size: int, steps: int):
    X = np.array(X)
    ks = [steps*i for i in range(1, (size//steps)+1)]
    X_temp = np.zeros(len(ks))

    for idx, k in enumerate(ks):
        X_temp[idx] = X[:k].mean()

    return X_temp


if __name__ == "__main__":
    simula_atendimento_wrapper(5, 3, .5, 50)

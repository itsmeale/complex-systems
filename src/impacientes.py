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


def intervalo_confianca(v: np.array, size: int) -> float:
    return 2 * 1.96 * std_error(v, size)


def std_error(v: np.array, size: int) -> float:
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

    X = np.zeros(simulations)
    Y = np.zeros(simulations)
    W = np.zeros(simulations)
    TM = np.zeros(simulations)
    LS_w = np.zeros(simulations//steps+1)
    LI_w = np.zeros(simulations//steps+1)
    LS_tm = np.zeros(simulations//steps+1)
    LI_tm = np.zeros(simulations//steps+1)

    while interv_confianca >= 0.0005:

        for i in range(steps):
            xi, yi, wi, tmi = simula_atendimento(n, lambd, mi, T)
            X[idx] = xi
            Y[idx] = yi
            W[idx] = wi
            TM[idx] = tmi
            idx += 1

        std_error_w = std_error(W, idx)
        std_error_tm = std_error(TM, idx)

        LS_w[idx_steps] = W.mean() + std_error_w
        LI_w[idx_steps] = W.mean() - std_error_w
        LS_tm[idx_steps] = TM.mean() + std_error_tm
        LI_tm[idx_steps] = TM.mean() - std_error_tm

        idx_steps += 1
        interv_confianca = intervalo_confianca(W, idx)
        print(f"{idx}: intervalo de confianca {intervalo_confianca(W, idx)/2}")

    n = idx
    intervalos = [LS_w, LI_w, LS_tm, LI_tm]
    simulacoes = [X, Y, W, TM]
    medias_moveis = [
        media_movel(X, n, steps),
        media_movel(Y, n, steps),
        media_movel(W, n, steps),
        media_movel(TM, n, steps)
    ]
    k_set = [steps*i for i in range(1, (n//steps)+1)]

    return simulacoes, intervalos, medias_moveis, k_set


def media_movel(X: np.array, size: int, steps: int):
    ks = [steps*i for i in range(1, (size//steps)+1)]
    X_temp = np.zeros(len(ks))

    for idx, k in enumerate(ks):
        X_temp[idx] = X[:k].mean()

    return X_temp


if __name__ == "__main__":
    # ux, uy, uw, utm = simula_atendimento_wrapper(5, 3, .5, 50)
    # print(f"media aceita: {ux}")
    # print(f"media rejeitada: {uy}")
    # print(f"proporcao de rejeitadas: {uw}")
    # print(f"tempo maximo (media): {utm}")
    pass

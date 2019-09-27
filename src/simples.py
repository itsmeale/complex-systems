# -*- coding: utf-8 -*-
from typing import List
import numpy as np
from random import expovariate


def simula_atendimento(
    n: int,
    lambd: int,
    th: int,
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

    th: int
        tempo de atendimento.

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
    tr = 0  # instante da ultima requisicao de entrada
    t = np.zeros(n)  # tempo de disponibilidade das linhas
    x, y = 0, 0  # aceitas e rejeitadas
    z = expovariate(lambd)

    while tr + z <= T:
        tr += z

        linha_disp = np.argmin(t)
        t_disp = t[linha_disp]

        if t_disp <= tr:
            t[linha_disp] = tr + th
            x += 1
        else:
            y += 1

        z = expovariate(lambd)

    return x, y, y/(x+y)


def simula_atendimento_wrapper(
    n: int,
    lambd: int,
    th: int,
    T: int,
    N: int,
) -> List[List[float]]:
    X = np.zeros(N)
    Y = np.zeros(N)
    W = np.zeros(N)

    for i in range(N):
        xi, yi, wi = simula_atendimento(n, lambd, th, T)
        X[i] = xi
        Y[i] = yi
        W[i] = wi

    return X.mean(), Y.mean(), W.mean()


if __name__ == "__main__":
    ux, uy, uw = simula_atendimento_wrapper(6, 2, 2, 50, 20000)
    print(f"media aceita: {ux}")
    print(f"media rejeitada: {uy}")
    print(f"proporcao de rejeitadas: {uw}")

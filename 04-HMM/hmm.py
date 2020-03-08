# Author: Kaituo Xu, Fan Yu
import numpy as np

def forward_algorithm(O, HMM_model):
    """HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)   # 时间步数
    N = len(pi)  # 状态数
    prob = 0.0
    # Begin Assignment
    alpha = np.zeros((T, N))

    # initial
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][color2id[O[0]]]

    # recursion
    for t in range(1, T):
        for i in range(N):
            alpha[t][i] = sum(alpha[t-1][j] * A[j][i] for j in range(N)) * B[i][color2id[O[t]]]

    # calculate P(O|pi, A, B)
    prob = np.sum(alpha, axis=1)[-1]
    # End Assignment
    return prob


def backward_algorithm(O, HMM_model):
    """HMM Backward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment
    beta = np.zeros((T, N))

    # initial
    beta[T - 1, :] = 1

    # recursion
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t][i] = sum( A[i][j] * B[j][color2id[O[t + 1]]] * beta[t + 1][j] for j in range(N))

    # calculate P(O|pi, A, B)
    prob = sum(pi[i] * B[i][color2id[O[0]]] * beta[0][i] for i in range(N))

    # End Assignment
    return prob
 

def Viterbi_algorithm(O, HMM_model):
    """Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    best_prob, best_path = 0.0, []
    # Begin Assignment
    delta = np.zeros((T, N))
    psi = np.zeros((T, N))

    # initial
    for i in range(N):
        delta[0][i] = pi[i] * B[i][color2id[O[0]]]
        psi[0][i] = 0

    # recursion
    for t in range(1, T):
        for i in range(N):
            delta[t][i] = max(delta[t-1][j] * A[j][i] for j in range(N)) * B[i][color2id[O[t]]]
            psi[t][i] = np.argmax(np.array([delta[t-1][j] * A[j][i] for j in range(N)]))

    # ending
    best_prob = np.max(delta[-1, :])
    i = np.argmax(delta[-1, :])
    best_path.append(i + 1)

    # backtrack
    for t in range(T-2, -1, -1):
        i = int(psi[t+1][i])
        best_path.append(i + 1)

    best_path.reverse()
    # End Assignment
    return best_prob, best_path


if __name__ == "__main__":
    color2id = { "RED": 0, "WHITE": 1 }
    # model parameters
    pi = [0.2, 0.4, 0.4]
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    # input
    observations = ("RED", "WHITE", "RED")
    HMM_model = (pi, A, B)
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
    print(observ_prob_forward)

    observ_prob_backward = backward_algorithm(observations, HMM_model)
    print(observ_prob_backward)

    best_prob, best_path = Viterbi_algorithm(observations, HMM_model) 
    print(best_prob, best_path)

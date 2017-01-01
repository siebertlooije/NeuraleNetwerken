# !/usr/bin/python3

import random
import numpy as np
import matplotlib.pyplot as plt

def generate_label(percent=50):
    return 1 if random.randrange(0, 100) > percent else -1


def generate_dataset(P, N):
    """
    :param P: Number of examples
    :param N: Number of dimensions for the feature
    :return: the random generated dataset
    """

    ID = []
    for i in range(0, P):
        eps = []
        for j in range(0, N):
            eps.append(random.gauss(0, 1))

        S = generate_label()
        ID.append([eps, S])
    return ID


def load_dataset():
    mat = scipy.io.loadmat('data3.mat')
    xi = np.transpose(mat['xi'])
    tau = mat['tau']
    return (xi, tau)


def check_E(E, c):
    for i in range(0, len(E)):
        if (E[i] <= c):
            return False
    return True


def Feed_forward(w1,w2,eps):
    return (np.tanh(np.dot(w1,eps)) + np.tanh(np.dot(w2,eps)))


#e = ((feedforward() - label)^2)/2
#w = w - learning_step * gradient(e)
def calculate_weights(w1,w2, IDs, example, learning_step):
    e = ((Feed_forward(w1,w2,example) - IDs[example])^2)*0.5
    w1_new = w1 - learning_step * np.gradient(w1)*e
    w2_new = w2 - learning_step * np.gradient(w2)*e
    return w1_new,w2_new


#E = 1/p * 1/2 for all examples do : (feed_forward(eps ) - label(eps))^2
def sto_gradient_descent(w1,w2,examples, IDs):
    sum_grad = 0
    for example in examples:
        sum_grad += (Feed_forward(w1,w2,example) - IDs(example))^2

    #0.5/len(examples) = (1/2)*(1/len(examples)
    final_grad = 0.5/len(examples)*sum_grad
    return final_grad


def select_random_example(len_examples):
    return np.random.randint(len_examples, size = 1)[0]


def seq_training(ID, P, n, N, c):
    """
    :param ID: the dataset
    :param P: the number of examples
    :param n: the number of epoches
    :return:
    """
    w1 = np.random.uniform(N)
    w2 = np.random.uniform(N)
    cost_function = 0
    for epoch in range(0, n):
        cost_function = sto_gradient_descent(w1,w2,P,ID)
        random_example = P[select_random_example(len(P))]
        w1,w2 = calculate_weights(w1,w2,ID,random_example,n)
    return cost_function


def plot_c():
    Qts = []
    N = 20  # Number of features
    nD = 50  # Number of generated dataset
    n = 100  # Number of epoch
    alphas = np.arange(0.75, 3.25, 0.25)
    for c in [-0.1, -0.05, 0, 0.05, 0.1]:
        Qts = []
        for alpha in alphas:
            P = int(alpha * N)  # Number of examples

            succes_counter = 0.0
            counter = 0.0
            for i in range(2, nD):
                ID = generate_dataset(P, N)
                w, succes = seq_training(ID, P, n, N, c)

                if (succes):
                    succes_counter += 1
                counter += 1
            print(float(succes_counter / counter))
            Qts.append(float(succes_counter / counter))

        plt.plot(alphas, Qts, label=str(c) + " C")
    plt.xlabel('Alpha')
    plt.ylabel('Succes rate')
    axes = plt.gca()
    axes.set_ylim([-0.2, 1.2])
    plt.legend()
    plt.show()


def plot_different_parameters():
    Qts = []
    N = 20  # Number of features
    nD = 50  # Number of generated dataset
    n = 100  # Number of epoch

    alphas = np.arange(0.75, 3.25, 0.25)
    for N in [10, 20, 50, 100, 200]:
        Qts = []
        for alpha in alphas:
            P = int(alpha * N)  # Number of examples

            succes_counter = 0.0
            counter = 0.0
            for i in range(2, nD):
                ID = generate_dataset(P, N)
                w, succes = seq_training(ID, P, n, N)

                if (succes):
                    succes_counter += 1
                counter += 1
            print(float(succes_counter / counter))
            Qts.append(float(succes_counter / counter))

        plt.plot(alphas, Qts, label=str(N) + " features")
    plt.legend()
    plt.ylabel('Succes rate')
    plt.xlabel('alpha')
    plt.title('Succes rate for different alphas and dimensions')
    plt.show()


def main():
    plot_c()


if __name__ == '__main__':
    plot_different_parameters()

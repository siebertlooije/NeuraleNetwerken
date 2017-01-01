#!/usr/bin/python3

import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# def check_E(E, c):
#     for i in range(0, len(E)):
#         if (E[i] <= c):
#             return False
#     return True


def generate_label(percent=50):
    return 1 if random.randrange(0, 100) > percent else -1


def load_dataset():
    mat = scipy.io.loadmat('data3.mat')
    xi = np.transpose(mat['xi'])
    tau = mat['tau']
    return (xi, tau)


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


def seq_training(ID, tau, P, n, N):
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
        cost_functions.append(cost_function)
    return cost_functions


def plot_different_parameters():
    P = 100 # Number of training samples
    Q = 100 # Number of test samples
    eta = 0.05 # Learning rate
    n = 100  # Number of epoch
    N = 50

    xi, tau = load_dataset()
    train_samples = xi[0:P] # First P examples for training
    test_samples = xi[P:P+Q] # Q examples after that for testing

    result = seq_training(train_samples, tau, P, n, N)



    # alphas = np.arange(0.75, 3.25, 0.25)
    # for alpha in alphas:
    #     P = int(alpha * N)  # Number of examples

    #     succes_counter = 0.0
    #     counter = 0.0
    #     for i in range(2, nD):
    #         ID = generate_dataset(P, N)
    #         w, succes = seq_training(ID, P, n, N)

    #         if (succes):
    #             succes_counter += 1
    #         counter += 1
    #     print(float(succes_counter / counter))
    #     Qts.append(float(succes_counter / counter))

    plt.plot(result)
    plt.ylabel(' ')
    plt.xlabel(' ')
    plt.title('Cost vs time')
    plt.show()


def main():
    plot_c()


if __name__ == '__main__':
    plot_different_parameters()

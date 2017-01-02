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
def calculate_weights(w1,w2, train_sample,train_tau, learning_step):
    e = np.sqrt(np.power(Feed_forward(w1,w2,train_sample) - train_tau,2))
    w1_new = w1 - learning_step * np.gradient(w1)*e
    w2_new = w2 - learning_step * np.gradient(w2)*e
    return w1_new,w2_new

#E = 1/p * 1/2 for all examples do : (feed_forward(eps ) - label(eps))^2
def sto_gradient_descent(w1,w2, samples, tau):
    sum_grad = 0
    for index,sample in enumerate(samples):
        result = np.power(Feed_forward(w1,w2,sample) - tau[index],2)
        sum_grad += result
    final_grad = 0.5/len(samples)*sum_grad
    return final_grad


def select_random_example(len_examples):
    return np.random.randint(len_examples, size = 1)[0]


def seq_training(train_samples, test_samples, train_tau,test_tau, P, n, N):
    """
    :param ID: the dataset
    :param P: the number of examples
    :param n: the number of epoches
    :return:
    """
    w1 = np.random.uniform(-1,1,N)
    w2 = np.random.uniform(-1,1,N)

    cost_function = 0
    cost_train = []
    cost_test = []
    for epoch in range(0, n):

        cost_tr = sto_gradient_descent(w1,w2,train_samples,train_tau)
        cost_train.append(cost_tr)
        cost_tst = sto_gradient_descent(w1,w2,test_samples,test_tau)
        cost_test.append(cost_tst)
        random_example = select_random_example(len(train_samples))
        train_sample = train_samples[random_example]
        random_tau = train_tau[random_example]
        w1,w2 = calculate_weights(w1,w2,train_sample,random_tau, n)
    return cost_train,cost_test


def plot_different_parameters():
    P = 100 # Number of training samples
    Q = 100 # Number of test samples
    eta = 0.05 # Learning rate
    n = 100  # Number of epoch
    N = 50

    xi, tau = load_dataset()
    tau_flat = tau[0]
    train_samples = xi[0:P] # First P examples for training
    train_tau = tau_flat[0:P]
    test_tau = tau_flat[P:P+Q]
    test_samples = xi[P:P+Q] # Q examples after that for testing
    print(train_samples.shape)
    print(test_samples.shape)
    print(tau_flat.shape)
    print(train_tau.shape)
    print(test_tau.shape)
    result_train,result_test = seq_training(train_samples,test_samples, train_tau,test_tau,  P, n, N)



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

    plt.plot(result_train)
    plt.plot(result_test)
    plt.ylabel(' ')
    plt.xlabel(' ')
    plt.title('Cost vs time')
    plt.show()


if __name__ == '__main__':
    plot_different_parameters()

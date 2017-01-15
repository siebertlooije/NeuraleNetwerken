#!/usr/bin/python3

import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
# def check_E(E, c):
#     for i in range(0, len(E)):
#         if (E[i] <= c):
#             return False
#     return True


def sigmoid(x):
    #http://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
    return 1 / (1 + np.exp(-x))

def generate_label(percent=50):
    return 1 if random.randrange(0, 100) > percent else -1


def load_dataset():
    mat = scipy.io.loadmat('data3.mat')
    xi = np.transpose(mat['xi'])
    tau = mat['tau']
    return (xi, tau)


def Feed_forward_sigmoid(w1,w2,eps):
    return (sigmoid(np.dot(w1,eps)) + sigmoid(np.dot(w2,eps)))

def Feed_forward(w1,w2,eps):
    return (np.tanh(np.dot(w1,eps)) + np.tanh(np.dot(w2,eps)))


#e = ((feedforward() - label)^2)/2
#w = w - learning_step * gradient(e)
def calculate_weights(w1,w2, train_sample,train_tau, learning_step, sigmoid = False):
    if sigmoid :
        e = np.sqrt(np.power(Feed_forward_sigmoid(w1,w2,train_sample) - train_tau,2))
    else:
        e = np.sqrt(np.power(Feed_forward(w1, w2, train_sample) - train_tau, 2))
    w1_new = w1 - learning_step * np.gradient(w1)*e
    w2_new = w2 - learning_step * np.gradient(w2)*e
    return w1_new,w2_new

#E = 1/p * 1/2 for all examples do : (feed_forward(eps ) - label(eps))^2
def sto_gradient_descent(w1,w2, samples, tau, sigmoid = False):
    sum_grad = 0
    for index,sample in enumerate(samples):
        if sigmoid :
            result = np.power(Feed_forward_sigmoid(w1,w2,sample) - tau[index],2)
        else :
            result = np.power(Feed_forward(w1,w2,sample) - tau[index],2)
        sum_grad += result
    final_grad = 0.5/len(samples)*sum_grad
    return final_grad


def select_random_example(len_examples):
    return np.random.randint(len_examples, size = 1)[0]


def seq_training(train_samples, test_samples, train_tau,test_tau, n, N, eta, sigmoid):
    """
    :param ID: the dataset
    :param P: the number of examples
    :param n: the number of epoches
    :return:
    """
    w1 = np.random.uniform(-1,1,N)
    w2 = np.random.uniform(-1,1,N)
    cost_train = []
    cost_test = []
    for epoch in range(0, n):
        cost_tr = sto_gradient_descent(w1,w2,train_samples,train_tau,sigmoid= sigmoid)
        cost_train.append(cost_tr)
        cost_tst = sto_gradient_descent(w1,w2,test_samples,test_tau, sigmoid=sigmoid)
        cost_test.append(cost_tst)
        random_example = select_random_example(len(train_samples))
        train_sample = train_samples[random_example]
        random_tau = train_tau[random_example]
        w1,w2 = calculate_weights(w1,w2,train_sample,random_tau, eta, sigmoid= sigmoid)
    return cost_train,cost_test, w1,w2


def plot_different_parameters():
    P = 300 # Number of training samples
    Q = 100 # Number of test samples
    eta = 0.05 # Learning rate
    n = 8000  # Number of epoch
    N = 50
    xi, tau = load_dataset()
    tau_flat = tau[0]
    train_samples = xi[0:P] # First P examples for training
    train_tau = tau_flat[0:P]
    test_tau = tau_flat[P:P+Q]
    test_samples = xi[P:P+Q] # Q examples after that for testing
    sigmoid = False

    result_train,result_test,w1,w2 = seq_training(train_samples,test_samples, train_tau,test_tau, n, N, eta, sigmoid)

    print(w1.shape)
    print(w2.shape)

    """
    ind = ind = np.arange(50)
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind,w1,width, color='r', label="Weight 1")
    rects1 = ax.bar(ind + width,w2,width, color='y', label="Weight 2")
    plt.legend()
    plt.ylabel("Weight value")
    plt.xlabel("Weight number")
    plt.title("Weights of the feedforward network")
    plt.show()
    plt.savefig("weights.png")
    """
    plt.plot(result_train, label="Training error  " + str("tanh"))
    plt.plot(result_test, label="Test error  " + str("tanh"))
    plt.legend(fontsize = 24)
    plt.ylabel('Cost function', fontsize=24)
    plt.xlabel('Number of epoch ',fontsize=24)
    plt.title('Feedforward neural network with gradient descent cost vs epoch',fontsize=24)
    plt.show()
    plt.savefig("Feedfoward1.png")




if __name__ == '__main__':
    plot_different_parameters()

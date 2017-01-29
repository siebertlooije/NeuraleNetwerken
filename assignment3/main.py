#!/usr/bin/python3

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import scipy.io

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

def Feed_forward(w1,w2,eps, w3 = None):
    return (np.tanh(np.dot(w1,eps)) + np.tanh(np.dot(w2,eps))) if w3 == None else (np.tanh(np.dot(w1,eps)) + np.tanh(np.dot(w2,eps))+ np.tanh(np.dot(w3,eps)))

#e = ((feedforward() - label)^2)/2
#w = w - learning_step * gradient(e)
def calculate_weights(w1,w2, train_sample,train_tau, learning_step, sigmoid = False, w3 = None):
    if sigmoid :
        e = np.sqrt(np.power(Feed_forward_sigmoid(w1,w2,train_sample) - train_tau,2))
    else:
        e = np.sqrt(np.power(Feed_forward(w1, w2, train_sample, w3) - train_tau, 2))

    w1_new = w1 - learning_step * np.gradient(w1)*e
    w2_new = w2 - learning_step * np.gradient(w2)*e
    if w3 != None:
        w3_new = w3 - learning_step * np.gradient(w3)*e
    else:
        w3_new = w3
    return w1_new,w2_new,w3_new

#E = 1/p * 1/2 for all examples do : (feed_forward(eps ) - label(eps))^2
def sto_gradient_descent(w1,w2, samples, tau, sigmoid = False, w3 = None):
    sum_grad = 0
    for index,sample in enumerate(samples):
        if sigmoid :
            result = np.power(Feed_forward_sigmoid(w1,w2,sample) - tau[index],2)
        else :
            result = np.power(Feed_forward(w1,w2,sample,w3) - tau[index],2)
        sum_grad += result
    final_grad = 0.5/len(samples)*sum_grad
    return final_grad


def select_random_example(len_examples):
    return np.random.randint(len_examples, size = 1)[0]


def seq_training(train_samples, test_samples, train_tau,test_tau, n, N, eta, sigmoid, three_weight = False):
    """
    :param ID: the dataset
    :param P: the number of examples
    :param n: the number of epoches
    :return:
    """
    w1 = np.random.uniform(-1,1,N)
    w2 = np.random.uniform(-1,1,N)
    if three_weight:
        w3 = np.random.uniform(-1,1,N)
    else:
        w3 = None
    cost_train = []
    cost_test = []
    for epoch in range(0, n):
        cost_tr = sto_gradient_descent(w1,w2,train_samples,train_tau,sigmoid= sigmoid,w3 = w3)
        cost_train.append(cost_tr)
        cost_tst = sto_gradient_descent(w1,w2,test_samples,test_tau, sigmoid=sigmoid,w3 = w3)
        cost_test.append(cost_tst)
        random_example = select_random_example(len(train_samples))
        train_sample = train_samples[random_example]
        random_tau = train_tau[random_example]
        w1,w2,w3 = calculate_weights(w1,w2,train_sample,random_tau, eta, sigmoid= sigmoid,w3 = w3)
    return cost_train,cost_test, w1,w2,w3


def plot_different_parameters():
    P = 2000 # Number of training samples
    Q = 100 # Number of test samples
    eta = 0.01 # Learning rate
    n = 4000  # Number of epoch
    N = 50
    xi, tau = load_dataset()
    tau_flat = tau[0]
    train_samples = xi[0:P] # First P examples for training
    train_tau = tau_flat[0:P]
    test_tau = tau_flat[P:P+Q]
    test_samples = xi[P:P+Q] # Q examples after that for testing
    sigmoid = False
    counter = 0
    array_train = []
    array_test = []

    for three_weight in [False,True]:
        result_train,result_test,w1,w2,w3 = seq_training(train_samples,test_samples, train_tau,test_tau, n, N, eta, sigmoid, three_weight = three_weight)

        print(w1.shape)
        print(w2.shape)
     #   print(w3.shape)

        ind = ind = np.arange(50)
        width = 0.175
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind,w1,width, color='r', label="Weight 1")
        rects1 = ax.bar(ind + width,w2,width, color='y', label="Weight 2")
        if three_weight:
            width2 = 0.175
            rects1 = ax.bar(ind + width + width2, w3, width, color='b', label="Weight 3")

        plt.legend()
        plt.ylabel("Weight value")
        plt.xlabel("Weight number")
        plt.title("Weights of the feedforward network")
        plt.show()
        array_train.append(result_train)
        array_test.append(result_test)
        print(three_weight)
    for index,result in enumerate(array_train):
        plt.plot(array_train[index], label="Training error " + str(index +2) +" weights")
        plt.plot(array_test[index], label="test error " + str(index +2) +" weights")
    plt.legend(fontsize = 24)
    plt.ylabel('Cost function', fontsize=24)
    plt.xlabel('Number of epoch ',fontsize=24)
    plt.title('Feedforward neural network with gradient descent cost vs epoch',fontsize=24)
    plt.show()
    plt.savefig("Feedfoward1.png")

if __name__ == '__main__':
    plot_different_parameters()

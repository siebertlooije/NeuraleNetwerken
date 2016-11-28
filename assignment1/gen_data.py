import random
import numpy as np


P = 2 #Number of examples
N = 5 #Number of features
n = 5 #Number of epoch

def generate_label(percent=50):
    return 1 if random.randrange(0, 100) > percent else -1

def generate_dataset(P, N):
    """

    :param P: Number of examples
    :param N: Number of dimensions for the feature
    :return: the random generated dataset
    """

    ID = []
    for i in range(0,P):

        eps = []
        for j in range(0,N):
            eps.append(random.gauss(0,1))

        S = generate_label()
        ID.append([eps,S])
    return ID


def check_E(E):
    for i in range(0,len(E)):
        if(E[i] <= 0):
            return False
    return True

#E = w * eps * S
#TODO: This function is not good yet
def RosenBlatt_algorithm(eps, S, N, weight):
    """
    :param eps:
    :param S:
    :param N:
    :param w:
    :return:
    """

    E = weight * eps * S

    for index,w in enumerate(weight):
        if w * eps * S <= 0:
            weight[index] = w + ((1/N)*eps *S)

    return weight,E


def seq_training(ID, P, n, N):
    """
    :param ID: the dataset
    :param P: the number of examples
    :param n: the number of epoches
    :return:
    """
    w = np.zeros(P)
    E = np.zeros(P)
    for epoch in range(0,n * P):
        if(check_E(E)):
            return w,E
        index =  epoch % P
        w,E[index] = RosenBlatt_algorithm(ID[index][0],ID[index][1],N,w)

    return w,E


for i in range(2,10):
    ID = generate_dataset(P,i)
    w,E = seq_training(ID,P,n,i)

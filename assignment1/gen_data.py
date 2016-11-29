import random
import numpy as np

alpha = 1.00
N = 20 #Number of features
P = int(alpha*N) #Number of examples
nD = 50 #Number of generated dataset
n = 100 #Number of epoch

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
def RosenBlatt_algorithm(eps, S, N, weight):
    """
    :param eps:
    :param S:
    :param N:
    :param w:
    :return:
    """
    E = np.dot(weight,eps) * S
    new_weight = []
    for index,w in enumerate(weight):
        if E <= 0:
            temp = w + (1/n)*eps[index]*S
            new_weight.append(temp)
        else:
            new_weight.append(w)
    return new_weight,E


def seq_training(ID, P, n, N):
    """
    :param ID: the dataset
    :param P: the number of examples
    :param n: the number of epoches
    :return:
    """
    w = np.zeros(N)
    E = np.zeros(N)
    for epoch in range(0,n * P):

        if(check_E(E)):
            return w,E, True

        index =  epoch % P
        w,E[index] = RosenBlatt_algorithm(ID[index][0],ID[index][1],N,w)

    return w,E, False

succes_counter = 0.0
counter = 0.0
for i in range(2,nD):
    ID = generate_dataset(P,N)
    w,E,succes = seq_training(ID,P,n,N)
    if (succes):
        succes_counter += 1
    counter += 1

Qts = float(succes_counter / counter)
print(Qts)
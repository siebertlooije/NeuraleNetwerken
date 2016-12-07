#!/usr/bin/python3

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
    for i in range(0,P):
        eps = []
        for j in range(0,N):
            eps.append(random.gauss(0,1))

        S = generate_label()
        ID.append([eps,S])
    return ID


def check_E(E, c):
    for i in range(0,len(E)):
        if(E[i] <= c):
            return False
    return True

#E = w * eps * S

def RosenBlatt_algorithm(example, N, weight):
    

    """
    :param eps:
    :param S:
    :param N:
    :param w:
    :return:
    """

    #E = np.dot(weight,example[0]) * example[1]

    new_weight = []

    E = np.dot(weight,example[0]) * example[1]
    if E <= 0:
        temp = [(1/N) * ex * example[1] for ex in example[0]]
        new_weight = weight + temp
    else:
        new_weight = weight
    return new_weight,E



def plot_c():
    Qts = []
    N = 20 #Number of features
    nD = 50 #Number of generated dataset
    n = 100 #Number of epoch
    alphas = np.arange(0.75,3.25,0.25)
    for c in [-0.1,-0.05,0,0.05,0.1]:
        Qts = []
        for alpha in alphas:
            P = int(alpha*N) #Number of examples
            
            succes_counter = 0.0
            counter = 0.0
            for i in range(2,nD):
                ID = generate_dataset(P,N)
                w,succes = seq_training(ID,P,n,N,c)

                if (succes):
                    succes_counter += 1
                counter += 1
            print(float(succes_counter/counter))
            Qts.append(float(succes_counter/counter))


        plt.plot(alphas,Qts,label=str(c)+" C")
    plt.xlabel('Alpha')
    plt.ylabel('Succes rate')
    axes = plt.gca()
    axes.set_ylim([-0.2,1.2])
    plt.legend()
    plt.show()

def seq_training(ID, P, n, N, c):
    """
    :param ID: the dataset
    :param P: the number of examples
    :param n: the number of epoches
    :return:
    """

    E = np.zeros(P)
    w = np.zeros(N)
    for epoch in range(0,n):
        for example in range(0,P):
            if(check_E(E,c)):
                return w, True

            w,E[example] = RosenBlatt_algorithm(ID[example],N,w)
    
    return w, False


def plot_different_parameters():
    Qts = []
    N = 20 #Number of features
    nD = 50 #Number of generated dataset
    n = 100 #Number of epoch

    alphas = np.arange(0.75,3.25,0.25)
    for N in [10,20,50,100,200]:
        Qts = []
        for alpha in alphas:
            P = int(alpha*N) #Number of examples
            
            
            succes_counter = 0.0
            counter = 0.0
            for i in range(2,nD):
                ID = generate_dataset(P,N)
                w,succes = seq_training(ID,P,n,N)

                if (succes):
                    succes_counter += 1
                counter += 1
            print(float(succes_counter/counter))
            Qts.append(float(succes_counter/counter))


        plt.plot(alphas,Qts,label=str(N) + " features")
    plt.legend()
    plt.ylabel('Succes rate')
    plt.xlabel('alpha')
    plt.title('Succes rate for different alphas and dimensions')
    plt.show()

def main():
    plot_c()

if __name__ == '__main__':
    plot_different_parameters()
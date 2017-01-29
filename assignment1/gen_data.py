#!/usr/bin/python3

import random
import numpy as np
import matplotlib.pyplot as plt
import math
def add_noise(lambda_perc=50):
    return 1 if random.randrange(0, 100) > lambda_perc else -1


def generate_eps(N):
    eps = []
    for j in range(0,N):
        eps.append(random.gauss(0,1))
    return eps

def generate_label(eps,N, noise=False, lambda_perc=50):
    w_start = np.ones(N)
    added_noise = 1 if (noise == False) else add_noise(lambda_perc=lambda_perc)
    return (added_noise * np.sign(np.dot(w_start, eps)))

def generate_dataset(P, N, lambda_perc):
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
        S = generate_label(eps,N, noise = False, lambda_perc=lambda_perc)
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
    alphas = np.arange(0.75,100.00,1.00)
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

def seq_training(ID, P, n, N,c):
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


def calculate_gen_error(weight, N):
    w_start = np.ones(N)
    return (1 / math.pi) * (np.arccos(np.dot(w_start, weight) / (np.linalg.norm(w_start) * np.linalg.norm(weight))))


def plot_different_parameters():
    N = 20  # Number of features
    nD = 20  # Number of generated dataset

    alphas = np.arange(0.25, 5.0, 0.25)
    criterionAngle = 0.01
    lambda_error = 0
    #for lambda_error in np.arange(0, 50, 10):

    for N in range(100,200,25):
        average_error = []
        for alpha in alphas:
            P = int(alpha * N)  # Number of examples
            n = P * N  # Number of epoch
            errors = []
            for i in range(2, nD):
                ID = generate_dataset(P, N, lambda_error)
                w, succes = seq_training(ID, P, n, N,0)
                errors.append(calculate_gen_error(w, N))
            average_error.append(np.mean(errors))
            print("ALPHA :{}, SUCCES :{}, ERROR :{}".format(alpha, succes, np.mean(errors)))
        plt.plot(alphas, average_error, label=str(N) +" Number of features")
    plt.legend(loc=2, prop={'size': 6})
    plt.ylabel('Generalization error')
    plt.xlabel('alpha')
    plt.title('Generalization error for different alphas and dimensions')
    plt.savefig("Plot different N b")

"""
def plot_different_parameters():
    Qts = []
    N = 20 #Number of features
    nD = 50 #Number of generated dataset
    n = 100 #Number of epoch

    alphas = np.arange(0.75,3.25,0.25)
    for lambda_error  in range(0,50,10):
    #for N in [10,20,50,100,200]:
        Qts = []
        for alpha in alphas:
            P = int(alpha*N) #Number of examples
            
            
            succes_counter = 0.0
            counter = 0.0
            for i in range(2,nD):
                ID = generate_dataset(P,N, lambda_error)
                w,succes = seq_training(ID,P,n,N, 0)

                if (succes):
                    succes_counter += 1
                counter += 1
            print(float(succes_counter/counter))
            Qts.append(float(succes_counter/counter))


        plt.plot(alphas,Qts,label=str(lambda_error) + " % noise")
    plt.legend()
    plt.ylabel('Succes rate')
    plt.xlabel('alpha')
    plt.title('Succes rate for different alphas and dimensions')
    plt.savefig("noise 1.png")
"""

def main():
    plot_different_parameters()

if __name__ == '__main__':
    main()
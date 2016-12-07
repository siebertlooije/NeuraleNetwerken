#!/usr/bin/python3

import random
import numpy as np
import matplotlib.pyplot as plt
import math


def generate_eps(N):
    eps = []
    for j in range(0,N):
        eps.append(random.gauss(0,1))
    return eps

def generate_label(N):
    eps = generate_eps(N)
    w_start = np.ones(N)
    return np.sign(np.dot(w_start, eps))

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

        S = generate_label(N)
        ID.append([eps,S])
    return ID


def check_E(E, c):
    for i in range(0,len(E)):
        if(E[i] <= c):
            return False
    return True

#E = w * eps * S

def Minover_algorithm(examples, N, weight):
    

    """
    :param eps:
    :param S:
    :param N:
    :param w:
    :return:
    """
    new_weight = []
    index_k = 0
    high_k = 999
    for index,example in enumerate(examples):
        if (np.any(weight)):
            K = (np.dot(weight,example[0]) * example[1]) / np.linalg.norm(weight)
        else:
            K = 0
        
        if high_k > K:
            high_k= K
            index_k = index
    print(index_k)
    print(examples[index_k])

    temp = [(1/N) * ex * examples[index_k][1] for ex in examples[index_k][0]]
    new_weight = weight + temp

    return new_weight




def seq_training(ID, P, n, N):
    """
    :param ID: the dataset
    :param P: the number of examples
    :param n: the number of epoches
    :return:
    """

    control_w = np.zeros((P,N))
    w = np.zeros(N)
    for epoch in range(0,n):        
        w = Minover_algorithm(ID,N,w)
        control_w = control_w[:-1]
        control_w =  np.append(control_w,w)
        if(not np.any(np.std(control_w, axis=0))):
            return w, True
    return w, False



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



def calculate_gen_error(weight, N):
    return(1/math.pi) * (math.acos(np.dot(w_start,weight) / (np.linalg.norm(w_start) * np.linalg.norm(weight))))

def plot_different_parameters():
    Qts = []
    N = 20 #Number of features
    nD = 50 #Number of generated dataset
    n = 20 #Number of epoch

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

    N = 20 #Number of features
    alpha = 0.75 
    P = int(alpha*N) #Number of examples
    nD = 50 #Number of generated dataset
    n = P * N #Number of epoch

    ID = generate_dataset(P,N)
    w,succes = seq_training(ID,P,n,N)
    print("w : {}".format(w))
    print("succes : {}".format(succes))
    #plot_different_parameters()






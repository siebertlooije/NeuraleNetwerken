#!/usr/bin/python3

import random
import numpy as np
import matplotlib.pyplot as plt
import math


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


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

    temp = [(1/N) * ex * examples[index_k][1] for ex in examples[index_k][0]]
    new_weight = weight + temp

    return new_weight


def seq_training(ID, P, n, N, criterionAngle):
    """
    :param ID: the dataset
    :param P: the number of examples
    :param n: the number of epoches
    :return:
    """
    #control_w = np.zeros(P)
    w = np.zeros(N)
    control_w = w
    for epoch in range(0,n):
        w = Minover_algorithm(ID,N,w)
        #print (angle_between(w, control_w))
        if np.array_equal(w, control_w) or angle_between(w, control_w) < criterionAngle:
            return w, True
        else:
            control_w = w
    return w, False


    #     control_w = np.roll(control_w,1) #TODO: From here until line 90 something is not yet right.
    #     control_w = list(control_w[:-1])
    #     control_w.append(np.std(w))
    #     if(np.std(control_w) == 0): 
    #         return w,True
    # return w, False


def calculate_gen_error(weight,N):
    w_start = np.ones(N)
    return(1/math.pi) * (np.arccos(np.dot(w_start,weight) / (np.linalg.norm(w_start) * np.linalg.norm(weight))))


def plot_different_parameters():

    N = 20 #Number of features
    nD = 50 #Number of generated dataset
    
    
    alphas = np.arange(0.25,3,0.25)
    for criterionAngle in np.arange(0.01,0.35,0.08):
        average_error = []
        for alpha in alphas:
            P = int(alpha*N) #Number of examples
            n = P * N #Number of epoch
            errors = []
            for i in range(2,nD):
                ID = generate_dataset(P,N)
                w,succes = seq_training(ID,P,n,N, criterionAngle)
                errors.append(calculate_gen_error(w,N))
            average_error.append(np.mean(errors))
            print("ALPHA :{}, SUCCES :{}, ERROR :{}".format(alpha,succes, np.mean(errors)))
        plt.plot(alphas,average_error,label=str(criterionAngle) + " critangle")
    plt.legend()
    plt.ylabel('Generalization error')
    plt.xlabel('alpha')
    plt.title('Generalization error for different alphas and dimensions')
    plt.show()


if __name__ == '__main__':
    plot_different_parameters()
    #plot_different_parameters()

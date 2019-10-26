import _pickle as cPickle
import gzip
import numpy as np

training_set = []

def activation(z):
    toR = []
    for el in z:
        toR.append(int(1/(1 + np.exp((-1)*el))))
        # if el >= 0:
        #     toR.append(1)
        # else:
        #     toR.append(0)
    toR = np.array(toR)
    return toR

def makeList(digit):
    lst = np.array([0,0,0,0,0,0,0,0,0,0])
    lst[digit] = 1
    lst = np.array(lst)
    return lst

def activation_sec(z):
    tor = [0,0,0,0,0,0,0,0,0,0]
    max1 = z[0]
    ind = 0
    for l in range(0,len(z)):
        if z[l] > max1:
            max1 = z[l]
            ind = l
    tor[ind] = 1
    tor = np.array(tor)
    return tor

def perceptron_Algorithm(weights,bias,set,test):
    learning_Rate = 0.0135
    allClassified = False
    #print(set[0][0])
    nrIterations = 30
    nr_Of_Matches = 0
    bias = np.array(bias)
    while allClassified == False and nrIterations > 0:
        nr_Of_Matches = 0
        allClassified = True
        for i in range(0,len(set[0])):
            t = makeList(set[1][i])
            x = np.array([set[0][i]])
            z = weights.dot(x.transpose()) + bias
            output = activation(z)
            weights = weights + (np.transpose([t - output]).dot(x) * learning_Rate)
            bias = bias + np.transpose([t - output]) * learning_Rate
            ok = 1
            for index in range(0,len(output)):
                if output[index] != t[index]:
                    allClassified = False
                    ok = 0
                    break
            if ok == 1:
                nr_Of_Matches += 1
        #print(nr_Of_Matches)
        nrIterations -= 1
    nr_Of_Matches = 0
    for i in range(0,len(test[0])):
        t = makeList(test[1][i])
        x = np.array([test[0][i]])
        z = weights.dot(x.transpose()) + bias
        output = activation_sec(z)
        ok = 1
        for index in range(0,len(output)):
            if output[index] != t[index]:
                ok = 0
                break
        if ok == 1:
            nr_Of_Matches += 1
    print(nr_Of_Matches)


def main():
    f= gzip.open('mnist.pkl.gz','rb')
    training_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    # print(train_set[1][144])
    f.close()
    weights = []
    for i in range(0,10):
        weights.append([])
        for j in range(0,784):
            weights[i].append(0)
    weights = np.array(weights)
    bias = [[0,0,0,0,0,0,0,0,0,0]]
    bias = np.transpose(bias)
    #print(len(test_set[0]))
    #print(np.transpose([training_set[0][0]]))
    #8755
    perceptron_Algorithm(weights,bias,training_set,test_set)


main()
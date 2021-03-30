import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def generateMultiringDataset(numberOfClasses,numberOfSamples):
    C = numberOfClasses
    N = numberOfSamples
    
    #Generates N samples from C ring-shaped 
    #class-conditional pdfs with equal priors
    
    #Randomly determine class labels for each sample
    thr = np.linspace(0.0, 1.0, num=C+1) #split [0,1] into C equal length intervals
    u = np.random.rand(1, N) # generate N samples uniformly random in [0,1]
    labels = np.zeros((1,N))
    
    for l in range(C):
        ind_l = np.where((thr[l]<u) & (u<=thr[l+1]))
        labels[ind_l] = np.tile(l,(1,len(ind_l[0])))
    
    # parameters of the Gamma pdf needed later
    a = [pow(i, 2.5) for i in list(range(1,C+1))]
    b = np.tile(1.7, C)
    
    #Generate data from appropriate rings
    #radius is drawn from Gamma(a,b), angle is uniform in [0,2pi]
    angle = 2*math.pi*np.random.rand(1,N)
    radius = np.zeros((1,N)) # reserve space
    
    for l in range(C):
        ind_l = np.where(labels==l)
        radius[ind_l] = np.random.gamma(a[l], b[l], len(ind_l[0]))
        
    data = np.vstack((np.multiply(radius,np.cos(angle)), np.multiply(radius,np.sin(angle))))
    
    colors = iter(cm.rainbow(np.linspace(0, 1, C)))
    plt.figure()
    for l in range(C):
        ind_l = np.where(labels==l)[1]
        plt.scatter(data[0,ind_l], data[1,ind_l], color=next(colors), s=1)
    
    return data,labels



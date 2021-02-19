import numpy as np 
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl 
import math
import random
from scipy.stats import multivariate_normal

def generate_data(mu_1, mu_2, mu_3a, mu_3b, cov_1, cov_2, cov_3a, cov_3b, cp_1, cp_2, cp_3):
    '''
    Generate 10000 samples according to a multivariate Gaussian probability density function, 
    keeping track of the true class labels for each sample. Includes the 4 dimensions plus a
    0 or 1 to track true class labels.

    Parameters
    ----------
    The mus, covs, and cps of the Gaussian distributions.

    Returns
    -------
    samples: numpy.array
        The generated sample data
    '''
    rng = default_rng()
    overall_size = 10000
    size_1 = 0
    size_2 = 0
    size_3a = 0
    size_3b = 0
    for i in range(0, overall_size) :
        r = random.random()
        if(r < cp_1):
            size_1 = size_1 + 1
        elif(r < cp_1+cp_2):
            size_2 = size_2 + 1
        elif(r < cp_1+cp_2+(cp_3/2)):
            size_3a = size_3a + 1
        else:
            size_3b = size_3b + 1
    samples_1 = rng.multivariate_normal(mean=mu_1, cov=cov_1, size=size_1)
    samples_1 = pd.DataFrame(samples_1, columns=['x','y','z'])
    samples_1['True Class Label'] = 1
    samples_2 = rng.multivariate_normal(mean=mu_2, cov=cov_2, size=size_2)
    samples_2 = pd.DataFrame(samples_2, columns=['x','y','z'])
    samples_2['True Class Label'] = 2
    samples_3a = rng.multivariate_normal(mean=mu_3a, cov=cov_3a, size=size_3a)
    samples_3a = pd.DataFrame(samples_3a, columns=['x','y','z'])
    samples_3a['True Class Label'] = 3
    samples_3b = rng.multivariate_normal(mean=mu_3b, cov=cov_3b, size=size_3b)
    samples_3b = pd.DataFrame(samples_3a, columns=['x','y','z'])
    samples_3b['True Class Label'] = 3
    samples   = samples_1.append([samples_2, samples_3a, samples_3b])
    return samples

def write_sample_data(samples, save_path):
    '''
    Saves the sample data.

    Parameters
    ----------
    samples: numpy.array
        The generated sample data
    save_path: string
        File name and path to save the sample data

    Returns
    -------
    None
    '''
    samples.to_csv(save_path)

def read_sample_data(save_path):
    '''
    Read the sample data. Helper function to read data in other functions.

    Parameters
    ----------
    save_path: string
        File containing the sample data

    Returns
    -------
    samples: pandas.DataFrame
        The generated sample data
    '''
    samples = pd.read_csv(save_path, index_col=0)
    return samples

def plot_samples(samples_path, save_path='samples_scatterplot.pdf'):
    '''
    Plots the four-dimensions of the samples taken from the distribution.

    Parameters
    ----------
    samples_path: string
        File containing the sample data

    Returns
    -------
    None
    '''
    samples = read_sample_data(save_path=samples_path)
    fig = plt.figure(figsize = (10, 7))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04, wspace=0)
    ax = plt.axes(projection ="3d")
    samples_1 = samples[samples['True Class Label']==1]
    samples_2 = samples[samples['True Class Label']==2]
    samples_3 = samples[samples['True Class Label']==3]
    x_1 = samples_1['x'].tolist()
    y_1 = samples_1['y'].tolist()
    z_1 = samples_1['z'].tolist()
    x_2 = samples_2['x'].tolist()
    y_2 = samples_2['y'].tolist()
    z_2 = samples_2['z'].tolist()
    x_3 = samples_3['x'].tolist()
    y_3 = samples_3['y'].tolist()
    z_3 = samples_3['z'].tolist()
    ax.scatter3D(x_1, y_1, z_1, label='1', marker='o')
    ax.scatter3D(x_2, y_2, z_2, label='2', marker='^')
    ax.scatter3D(x_3, y_3, z_3, label='3', marker='s')
    ax.set_title("Samples from Multivariate Gaussian Distributions")
    ax.set_xlabel('1st Dimension, x')
    ax.set_ylabel('2nd Dimension, y')
    ax.set_zlabel('3rd Dimension, z')
    ax.legend()
    plt.savefig(save_path)
    plt.clf()
    return None

if __name__=='__main__':
    # Class 1
    cp_1 = 0.3
    # First Gaussian
    mu_1  = [1,1,1]
    cov_1 = [[0.5, 0,   1  ],
             [0,   2,   0  ],
             [0,   0.5, 0.5]]
    # Class 2
    cp_2 = 0.3
    # Second Guassian
    mu_2  = [1,1,2]
    cov_2 = [[1,   0,   1  ],
             [0,   1,   0  ],
             [0,   0.5, 1  ]]
    # Class 3
    cp_3 = 0.4
    # Third Gaussian
    mu_3a  = [1,2,2]
    cov_3a = [[3,   0,   0  ],
              [0,   1,   0  ],
              [0,   0,   0.5]]
    # Fourth Gaussian
    mu_3b  = [1,2,1]
    cov_3b = [[1,   0,   1  ],
              [0,   1,   0  ],
              [0,   0,   2  ]]

    samples_path = './samples_a.csv'
    
    samples = generate_data(mu_1, mu_2, mu_3a, mu_3b, cov_1, cov_2, cov_3a, cov_3b, cp_1, cp_2, cp_3)
    print(samples)
    write_sample_data(samples, samples_path)
    plot_samples(samples_path, save_path='samples_scatterplot_a.pdf')
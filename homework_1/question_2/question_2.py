import numpy as np 
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl 
import math
import random
from scipy.stats import multivariate_normal
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
    fig = plt.figure(figsize = (5,5))
    fig.subplots_adjust(left=0.01, right=0.985, top=0.99, bottom=0.01, wspace=0)
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
    ax.scatter3D(x_1, y_1, z_1, label='1', marker='o', alpha=0.2)
    ax.scatter3D(x_2, y_2, z_2, label='2', marker='^', alpha=0.2)
    ax.scatter3D(x_3, y_3, z_3, label='3', marker='s', alpha=0.2)
    #ax.set_title("Samples from Multivariate Gaussian Distributions")
    ax.set_xlabel('1st Dimension, x')
    ax.set_ylabel('2nd Dimension, y')
    ax.set_zlabel('3rd Dimension, z')
    ax.legend(loc='upper left', title='Class Label')
    plt.savefig(save_path)
    plt.clf()
    return None

def make_decisions(samples_path, sample_info, loss_matrix=[[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]):
    '''
    Implement classifier and check if correct given the true 
    data dsitribution knowledge. Chooses minimum risk.

    Parameters
    ----------
    samples: pandas.DataFrame
    sample_indo: pandas.DataFrame
    loss_matrix: array
        2d array = lambda

    Returns
    -------
    samples: pandas.DataFrame
        modified samples with classification and accuracy info.
    '''
    samples = read_sample_data(save_path=samples_path)
    choices  = []
    correct = []
    for idx, row in samples.iterrows():
        # Modify class label for computation
        distribution = int(row['True Class Label'])
        choice = np.argmin([risk(0,[row['x'],row['y'],row['z']],loss_matrix,sample_info), risk(1,[row['x'],row['y'],row['z']],loss_matrix,sample_info), 
                            risk(2,[row['x'],row['y'],row['z']],loss_matrix,sample_info), risk(3,[row['x'],row['y'],row['z']],loss_matrix,sample_info)])
        # Make sure 3a and 3b are together
        if(choice==0):
            choices.append(1) 
            choice = 1
        elif(choice==1):
            choices.append(2)
            choice = 2
        else:
            choices.append(3)
            choice = 3
        # Check if classification was correct or not
        if(choice==distribution):
            correct.append(True)
        else:
            correct.append(False)
    samples['ERM Classification'] = choices
    samples['Correct']            = correct
    return samples

def risk(i , x , loss_matrix, sample_info):
    '''
    Parameters
    ----------
    sample_info: pandas.DataFrame
        Info on true classes in distributions.
    i: int
        The true class assigned to i
    x: 
    p: float32
        The class prior
    loss_matrix: array, optional

    '''
    risk = 0
    for j, row in sample_info.iterrows():
        #  Probability, mu, sigma^2
        #print(j)
        if(i==j):
            continue
        #print(loss_matrix[i][j])
        print(multivariate_normal.pdf(x,row['mu'],row['cov']))
        risk = risk + loss_matrix[i][j]*row['P']*multivariate_normal.pdf(x,row['mu'],row['cov'])
        #print(risk)
    return risk

def plot_classified_samples(samples_path, save_path='samples_scatterplot.pdf'):
    '''
    Plots the four-dimensions of the classifications of the samples taken from the distribution.

    Parameters
    ----------
    samples_path: string
        File containing the sample data

    Returns
    -------
    None
    '''
    samples = read_sample_data(save_path=samples_path)
    fig = plt.figure(figsize = (5, 5))
    fig.subplots_adjust(left=0.01, right=0.985, top=0.99, bottom=0.01, wspace=0)
    ax = plt.axes(projection ="3d")
    samples_1 = samples[samples['ERM Classification']==1]
    samples_2 = samples[samples['ERM Classification']==2]
    samples_3 = samples[samples['ERM Classification']==3]
    x_1 = samples_1['x'].tolist()
    y_1 = samples_1['y'].tolist()
    z_1 = samples_1['z'].tolist()
    x_2 = samples_2['x'].tolist()
    y_2 = samples_2['y'].tolist()
    z_2 = samples_2['z'].tolist()
    x_3 = samples_3['x'].tolist()
    y_3 = samples_3['y'].tolist()
    z_3 = samples_3['z'].tolist()
    ax.scatter3D(x_1, y_1, z_1, label='1', marker='o', alpha=0.2)
    ax.scatter3D(x_2, y_2, z_2, label='2', marker='^', alpha=0.2)
    ax.scatter3D(x_3, y_3, z_3, label='3', marker='s', alpha=0.2)
    #ax.set_title("Samples from Multivariate Gaussian Distributions")
    ax.set_xlabel('1st Dimension, x')
    ax.set_ylabel('2nd Dimension, y')
    ax.set_zlabel('3rd Dimension, z')
    ax.legend(loc='upper left', title='Class Label')
    plt.savefig(save_path)
    plt.clf()
    return None

def plot_correct_classified_samples(samples_path, save_path='samples_classified_scatterplot.pdf'):
    '''
    Plots the four-dimensions of the samples taken from the distribution and if the classification was correct.

    Parameters
    ----------
    samples_path: string
        File containing the sample data

    Returns
    -------
    None
    '''
    samples = read_sample_data(save_path=samples_path)
    fig = plt.figure(figsize = (5,5))
    fig.subplots_adjust(left=0.01, right=0.985, top=0.99, bottom=0.01, wspace=0)
    ax = plt.axes(projection ="3d")
    # Plot correct
    correct = samples[samples['Correct']==True]
    samples_1 = correct[correct['True Class Label']==1]
    samples_2 = correct[correct['True Class Label']==2]
    samples_3 = correct[correct['True Class Label']==3]
    x_1 = samples_1['x'].tolist()
    y_1 = samples_1['y'].tolist()
    z_1 = samples_1['z'].tolist()
    x_2 = samples_2['x'].tolist()
    y_2 = samples_2['y'].tolist()
    z_2 = samples_2['z'].tolist()
    x_3 = samples_3['x'].tolist()
    y_3 = samples_3['y'].tolist()
    z_3 = samples_3['z'].tolist()
    ax.scatter3D(x_1, y_1, z_1, label='1', marker='o', alpha=0.2, color='green')
    ax.scatter3D(x_2, y_2, z_2, label='2', marker='^', alpha=0.2, color='green')
    ax.scatter3D(x_3, y_3, z_3, label='3', marker='s', alpha=0.2, color='green')
    # Plot incorrect
    correct = samples[samples['Correct']==False]
    samples_1 = correct[correct['True Class Label']==1]
    samples_2 = correct[correct['True Class Label']==2]
    samples_3 = correct[correct['True Class Label']==3]
    x_1 = samples_1['x'].tolist()
    y_1 = samples_1['y'].tolist()
    z_1 = samples_1['z'].tolist()
    x_2 = samples_2['x'].tolist()
    y_2 = samples_2['y'].tolist()
    z_2 = samples_2['z'].tolist()
    x_3 = samples_3['x'].tolist()
    y_3 = samples_3['y'].tolist()
    z_3 = samples_3['z'].tolist()
    ax.scatter3D(x_1, y_1, z_1, label='1', marker='o', alpha=0.2, color='red')
    ax.scatter3D(x_2, y_2, z_2, label='2', marker='^', alpha=0.2, color='red')
    ax.scatter3D(x_3, y_3, z_3, label='3', marker='s', alpha=0.2, color='red')
    ax.set_xlabel('1st Dimension, x')
    ax.set_ylabel('2nd Dimension, y')
    ax.set_zlabel('3rd Dimension, z')
    #ax.get_legend().remove()
    green_patch = mpatches.Patch(color='green', label='Correct')
    red_patch = mpatches.Patch(color='red', label='Incorrect')
    ax.legend(handles=[green_patch, red_patch], loc='upper left', title='Classification')
    plt.savefig(save_path)
    plt.clf()
    return None

def plot_decision_matrix(samples_path, save_path='./decision_matrix.pdf'):
    '''
    Plots a heatmap of the decision matrix along with the values.
    Parameters
    ----------
    samples_path: string
        File containing the sample data

    Returns
    -------
    None
    '''
    samples = read_sample_data(save_path=samples_path)
    pred = samples['ERM Classification'].tolist()
    act  = samples['True Class Label'].tolist()
    confusion = confusion_matrix(act, pred, normalize='true')
    print(confusion)
    sns.heatmap(data=confusion,cmap="YlOrRd",annot=True,)
    plt.xlabel('Decision')
    plt.ylabel('True Class Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    plt.close()

if __name__=='__main__':
    sample_info = pd.DataFrame(columns=['P','mu','cov'])
    # Class 1
    cp_1 = 0.3
    # First Gaussian
    mu_1  = [1,1,1]
    cov_1 = [[0.5, 0,   1  ],
             [0,   2,   0  ],
             [0,   0.5, 0.5]]
    d = {'P':cp_1,'mu':mu_1,'cov':cov_1}
    sample_info = sample_info.append(d,ignore_index=True)
    # Class 2
    cp_2 = 0.3
    # Second Guassian
    mu_2  = [1,1,2]
    cov_2 = [[1,   0,   1  ],
             [0,   1,   0  ],
             [0,   0.5, 1  ]]
    d = {'P':cp_2,'mu':mu_2,'cov':cov_2}
    sample_info = sample_info.append(d,ignore_index=True)
    # Class 3
    cp_3 = 0.4
    # Third Gaussian
    mu_3a  = [1,2,2]
    cov_3a = [[3,   0,   0  ],
              [0,   1,   0  ],
              [0,   0,   0.5]]
    d = {'P':(cp_3/2),'mu':mu_3a,'cov':cov_3a}
    sample_info = sample_info.append(d,ignore_index=True)
    # Fourth Gaussian
    mu_3b  = [1,2,1]
    cov_3b = [[1,   0,   1  ],
              [0,   1,   0  ],
              [0,   0,   2  ]]
    d = {'P':(cp_3/2),'mu':mu_3b,'cov':cov_3b}
    sample_info = sample_info.append(d,ignore_index=True)

    loss_matrix_10 = [[0,   1,   10  , 10],
                      [1,   0,   10  , 10],
                      [1,   1,   0   , 0 ],
                      [1,   1,   0   , 0 ]]
    loss_matrix_100 = [[0,   1,  100  , 100],
                       [1,   0,  100  , 100],
                       [1,   1,  0    , 0  ],
                       [1,   1,  0    , 0  ]]

    samples_path = './samples_a.csv'
    samples_b_10  = './samples_b_10.csv'
    samples_b_100 = './samples_b_100.csv'
    #samples = generate_data(mu_1, mu_2, mu_3a, mu_3b, cov_1, cov_2, cov_3a, cov_3b, cp_1, cp_2, cp_3)
    #write_sample_data(samples, samples_path)
    #plot_samples(samples_path, save_path='samples_scatterplot_a.pdf')
    samples = make_decisions(samples_path, sample_info)
    #write_sample_data(samples, samples_path)
    #plot_classified_samples(samples_path, save_path='samples_classified_scatterplot.pdf')
    #plot_correct_classified_samples(samples_path, save_path='samples_correct_classified_scatterplot.pdf')

    # Part B
    #samples_10 = make_decisions(samples_path, sample_info, loss_matrix=loss_matrix_10)
    #write_sample_data(samples_10, samples_b_10)
    #plot_classified_samples(samples_b_10, save_path='samples_classified_scatterplot_10.pdf')
    #plot_correct_classified_samples(samples_b_10, save_path='samples_correct_classified_scatterplot_10.pdf')
    #samples_100 = make_decisions(samples_path, sample_info, loss_matrix=loss_matrix_100)
    #write_sample_data(samples_100, samples_b_100)
    #plot_classified_samples(samples_b_100, save_path='samples_classified_scatterplot_100.pdf')
    #plot_correct_classified_samples(samples_b_100, save_path='samples_correct_classified_scatterplot_100.pdf')
    #plot_decision_matrix(samples_path, save_path='./decision_matrix.pdf')
    #plot_decision_matrix(samples_b_10, save_path='./decision_matrix_10.pdf')
    #plot_decision_matrix(samples_b_100, save_path='./decision_matrix_100.pdf')

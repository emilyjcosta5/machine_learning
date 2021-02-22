import numpy as np 
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl 
import math
import random
from scipy.stats import multivariate_normal

def generate_data(mean_0, mean_1, cov_0, cov_1):
    '''
    Generate 10000 samples according to a multivariate Gaussian probability density function, 
    keeping track of the true class labels for each sample. Includes the 4 dimensions plus a
    0 or 1 to track true class labels.

    Parameters
    ----------
    None

    Returns
    -------
    samples: numpy.array
        The generated sample data
    '''
    rng = default_rng()
    overall_size = 10000
    p_0 = 0.7
    p_1 = 0.3
    size_0 = 0
    size_1 = 0
    for i in range(0, overall_size) :
        if(random.random() < p_0):
            size_0 = size_0 + 1
        else:
            size_1 = size_1 + 1

    samples_0 = rng.multivariate_normal(mean=mean_0, cov=cov_0, size=size_0)
    samples_0 = pd.DataFrame(samples_0, columns=['x','y','z','t'])
    samples_0['True Class Label'] = 0
    samples_1 = rng.multivariate_normal(mean=mean_1, cov=cov_1, size=size_1)
    samples_1 = pd.DataFrame(samples_1, columns=['x','y','z','t'])
    samples_1['True Class Label'] = 1
    samples   = samples_0.append(samples_1)
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
    samples = samples.sort_values('t')
    samples_0 = samples[samples['True Class Label']==0]
    samples_1 = samples[samples['True Class Label']==1]
    x_0 = samples_0['x'].tolist()
    y_0 = samples_0['y'].tolist()
    z_0 = samples_0['z'].tolist()
    t_0 = samples_0['t'].tolist()
    x_1 = samples_1['x'].tolist()
    y_1 = samples_1['y'].tolist()
    z_1 = samples_1['z'].tolist()
    t_1 = samples_1['t'].tolist()
    C = np.linspace(-5, 5, len(t_0))
    scamap = plt.cm.ScalarMappable(cmap='YlOrRd')
    fcolors = scamap.to_rgba(C)
    ax.scatter3D(x_0, y_0, z_0, facecolors=fcolors, cmap='YlOrRd', label='0')
    C = np.linspace(-5, 5, len(t_1))
    scamap = plt.cm.ScalarMappable(cmap='PuBuGn')
    fcolors = scamap.to_rgba(C)
    ax.scatter3D(x_1, y_1, z_1, facecolors=fcolors, cmap='PuBuGn', label='1')
    ax.set_title("Samples from Multivariate Gaussian Distributions")
    ax.set_xlabel('1st Dimension, x')
    ax.set_ylabel('2nd Dimension, y')
    ax.set_zlabel('3rd Dimension, z')
    plt.savefig(save_path)
    plt.clf()
    
    # Make colorbar 
    fig, axes = plt.subplots(1,2,figsize=(8,1.2))
    fig.suptitle('True Class Labels')
    fig.text(0.5, 0.08, '4th Dimension, t', ha='center', va='center')
    cmap = plt.cm.get_cmap('YlOrRd')
    colors = cmap(np.arange(cmap.N))
    axes[0].imshow([colors], extent=[min(t_0), max(t_0), 0, 1])
    axes[0].set_yticklabels([])
    axes[0].set_yticks([])
    axes[0].set_ylim(0,1)
    axes[0].set_xlim(math.ceil(min(t_0)),math.floor(max(t_0)))
    axes[0].set_title('Class 0')

    cmap = plt.cm.get_cmap('PuBuGn')
    colors = cmap(np.arange(cmap.N))
    axes[1].imshow([colors], extent=[min(t_1), max(t_1), 0, 1])
    axes[1].set_yticklabels([])
    axes[1].set_yticks([])
    axes[1].set_ylim(0,1)
    axes[1].set_xlim(math.ceil(min(t_1)),math.floor(max(t_1)))
    axes[1].set_title('Class 1')

    plt.savefig('samples_scatterplot_legend.pdf')
    plt.clf()
    return None

def implement_classifier_and_plot_roc_curve(samples_path, mean_0, mean_1, cov_0, cov_1, save_path='./ROC_curve.pdf'):
    '''
    Plots the minimum risk and ROC curve with theorectical and experimental probabilites.

    Parameters
    ----------
    samples_path: string
        File containing the sample data
    mean_0, mean_1: array
        The vectors of mu for the two classes.
    cov_0, cov_1: double array
        The matrixes of sigma for the two classes.

    Returns
    -------
    exp_min: dict
        Info of the experimental minimum error.
    thy_min: dict
        Info of the theorectical minimum error.
    '''
    # Prep the sample data
    samples   = read_sample_data(save_path=samples_path)
    # Initialize empty arrays to store discriminant scores
    discriminants = []
    # Fill array with calculated discriminants and add to DataFrame
    for i in range(0, samples.shape[0]):
        sample = samples.iloc[i].to_numpy()[:-1]
        discriminant = multivariate_normal.pdf(sample, mean_1, cov_1)/multivariate_normal.pdf(sample, mean_0, cov_0)
        discriminants.append(discriminant)
    samples['Discriminant'] = discriminants
    samples = samples.sort_values('Discriminant')
    dis_0 = samples[samples['True Class Label']==0]['Discriminant'].tolist()
    dis_1 = samples[samples['True Class Label']==1]['Discriminant'].tolist()
    df = pd.DataFrame(columns=['False Positive', 'True Positive', 'Gamma', 'Probability Error'])
    for index, row in samples.iterrows():
        discriminant   = row['Discriminant'] 
        false_positive = len([class_dis for class_dis in dis_0 if class_dis>=discriminant])/len(dis_0)
        true_positive = len([class_dis for class_dis in dis_1 if class_dis>=discriminant])/len(dis_1)
        p_err = false_positive*0.7+(1-true_positive)*0.3
        d = {'False Positive': false_positive, 'True Positive': true_positive, 
             'Gamma': discriminant, 'Probability Error': p_err}
        df = df.append(d, ignore_index=True)
    df = df.sort_values('Probability Error')
    # Get info of minimum experimental probablility error
    exp_min = df.iloc[0]
    print('Experimental Mimimum Error Info:\n')
    print(exp_min)
    # Calculate theorectical error
    thy_gamma = 0.7/0.3
    thy_lambdas = [len([class_dis for class_dis in dis_0 if class_dis>=thy_gamma])/len(dis_0),
                len([class_dis for class_dis in dis_1 if class_dis>=thy_gamma])/len(dis_1)]
    thy_p_err = thy_lambdas[0]*0.7 + (1-thy_lambdas[1])*0.3
    thy_min = {'False Positive': thy_lambdas[0], 'True Positive': thy_lambdas[1], 'Gamma': thy_gamma, 'Probability Error': thy_p_err}
    print('Theoretical Mimimum Error Info:\n')
    print(thy_min)
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    # Plot ROC curve
    ax.plot(df['False Positive'], df['True Positive'], 'ro', markersize=4)
    # Plot experimental minimum
    ax.plot(exp_min['False Positive'], exp_min['True Positive'], 'bo', label='Experimental', markersize=10)
    # Plot theorectical minimum
    ax.plot(thy_min['False Positive'], thy_min['True Positive'], 'go', label='Theoretical', markersize=10)
    ax.legend(title='Minimum Error Probabilities', loc='lower right')
    #ax.set_title('Minimum Expected Risk ROC Curve')
    ax.set_xlabel('Probability of False Positive')
    ax.set_ylabel('Probability of True Positive')
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.xaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    plt.savefig(save_path)
    return exp_min, thy_min 

if __name__=='__main__':
    samples_path = './question_one_samples.csv'
    mean_0 = [-1, 1, -1, 1]
    mean_1 = [1, 1, 1, 1]
    cov_0  = [[2,    -0.5,  0.3,  0],
            [-0.5, 1,     -0.5, 0],
            [0.3,  -0.5,  1,    0],
            [0,    0,     0,    2]]
    cov_1  = [[1,     0.3,  -0.2, 0],
            [0.3,   2,    0.3,  0],
            [-0.2,  0.3,  1,    0],
            [0,     0,    0,    3]]
    incorrect_samples_path = './question_one_incorrect_sample.csv'
    cov_0_incorrect  = [[2,    0,     0,    0],
                        [0,    1,     0,    0],
                        [0,    0,     1,    0],
                        [0,    0,     0,    2]]
    cov_1_incorrect  = [[1,     0,    0,    0],
                        [0,     2,    0,    0],
                        [0,     0,    1,    0],
                        [0,     0,    0,    3]]
    # Part A
    '''
    samples = generate_data(mean_0, mean_1, cov_0, cov_1)
    write_sample_data(samples=samples, save_path=samples_path)
    plot_samples(samples_path=samples_path)
    implement_classifier_and_plot_roc_curve(samples_path, mean_0, mean_1, cov_0, cov_1)
    '''
    # Part B
    '''
    samples = generate_data(mean_0, mean_1, cov_0_incorrect, cov_1_incorrect)
    write_sample_data(samples=samples, save_path=incorrect_samples_path)
    plot_samples(samples_path=incorrect_samples_path, save_path='incorrect_samples_scatterplot.pdf')
    '''
    implement_classifier_and_plot_roc_curve(incorrect_samples_path, mean_0, mean_1, cov_0_incorrect, cov_1_incorrect, save_path='./incorrect_ROC_curve.pdf')
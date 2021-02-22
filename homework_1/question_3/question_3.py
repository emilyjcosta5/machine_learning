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

def estimate_cov_mu(data):
    '''
    Estimates the covariance matrix and mean vector for all the true class
    labels in the input data.

    Parameters
    ----------
    data: pandas.DataFrame
        Given data set

    Returns
    -------
    data_info: pandas.DataFrame
        The true labels, covariance matrics and mean vectors
    '''
    true_labels = data.index.unique().tolist()
    data_info   = pd.DataFrame(columns=['True Class Label', 'Covariance Matrix', 'Mean Vector', 'Number of Samples', 'Class Prior'])
    total_samples = 0
    for true_label in true_labels:
        temp = data.loc[true_label, :]
        #cov = np.cov(temp, bias=True)
        cov  = temp.cov().to_numpy()
        mean = temp.mean(axis=0).tolist()
        n = temp.shape[0]
        total_samples = total_samples + n
        d = {'True Class Label': true_label, 'Covariance Matrix': cov, 'Mean Vector': mean, 'Number of Samples': n}
        data_info = data_info.append(d, ignore_index=True)
    data_info['Class Prior'] = data_info['Number of Samples'] / total_samples
    return data_info

def plot_subset(data, subset=['x','y','z']):
    '''
    Plots the four-dimensions of the samples taken from the distribution.

    Parameters
    ----------
    data: pandas.DataFrame
        Contains the sample data
    subset: array, optional
        Plots these three values.

    Returns
    -------
    None
    '''
    markers = ['v', '^', '<', '>', '8', 's', 'p', '*', 'h', '+', 'x', 'D']
    fig = plt.figure(figsize = (5.5,5))
    fig.subplots_adjust(left=0.01, right=0.96, top=0.99, bottom=0.01, wspace=0)
    ax = plt.axes(projection ="3d")
    true_labels = data.sort_index().index.unique().tolist()
    for i, true_label in enumerate(true_labels):
        temp = data.loc[true_label, :]
        xs = temp[subset[0]].tolist()
        ys = temp[subset[1]].tolist()
        zs = temp[subset[2]].tolist()
        ax.scatter3D(xs, ys, zs, label=true_label, marker=markers[i], alpha=0.3)
    ax.set_xlabel('%s'%subset[0])
    ax.set_ylabel('%s'%subset[1])
    ax.set_zlabel('%s'%subset[2])
    ax.legend(loc='upper left', title='Class Label')
    #plt.tight_layout()
    plt.savefig('./%s_%s_%s_true_classes.pdf'%(subset[0], subset[1], subset[2]))
    plt.clf()
    return None

def make_decisions(data, data_info, loss_matrix=None, true_class_label='True Class Label'):
    '''
    Implement classifier and check if correct given the true 
    data dsitribution knowledge. Chooses minimum risk.

    Parameters
    ----------
    data: pandas.DataFrame
        Contains the sample data
    data_info: pandas.DataFrame
    loss_matrix: array
        2d array = lambda

    Returns
    -------
    data: pandas.DataFrame
        modified data with classification and accuracy info.
    '''
    choices  = []
    correct = []
    dimension_labels = data.columns.tolist()
    class_labels     = data.sort_index().index.unique().tolist()
    # Create 0-1 loss matrix if none is given
    if(loss_matrix==None):
        d = max(class_labels)
        loss_matrix = np.zeros((d,d))
        for i in range(0,d):
            for j in range(0,d):
                if(i==j):
                    loss_matrix[i][j] = 0
                else:
                    loss_matrix[i][j] = 1
    labels_reference  = {i:class_labels[i] for i in range(0,len(class_labels))}
    for idx, row in data.iterrows():
        # Modify class label for computation
        distribution = int(row.name)
        rows         = [row[dimension_label] for dimension_label in dimension_labels]
        args         = [risk(class_label-1, rows, loss_matrix, data_info) for class_label in class_labels]
        choice = labels_reference[np.argmin(args)]
        choices.append(choice)
        # Check if classification was correct or not
        if(choice==distribution):
            correct.append(True)
        else:
            correct.append(False)
    data['ERM Classification'] = choices
    data['Correct']            = correct
    return data

def risk(i , x , loss_matrix, data_info):
    '''
    Parameters
    ----------
    data_info: pandas.DataFrame
        Info on true classes in distributions.
    i: int
        The true class assigned to i
    x: 
    p: float32
        The class prior
    loss_matrix: array, optional

    '''
    risk = 0
    for j, row in data_info.iterrows():
        #  Probability, mu, sigma^2
        try:
            risk = risk + loss_matrix[i][int(row['True Class Label'])-1]*row['Class Prior']*multivariate_normal.pdf(x,row['Mean Vector'],row['Covariance Matrix'])
        except np.linalg.LinAlgError:
            continue
    return risk

def plot_decision_matrix(data, save_path='./decision_matrix.pdf', true_class_label='True Class Label'):
    '''
    Plots a heatmap of the decision matrix along with the values.
    Parameters
    ----------
    data: pandas.DataFrame
        Contains the sample data

    Returns
    -------
    None
    '''
    pred = data['ERM Classification'].tolist()
    act  = data.index.tolist()
    class_labels = data.sort_index().index.unique().tolist()
    confusion = confusion_matrix(act, pred, normalize='true')
    sns.heatmap(data=confusion,cmap="YlOrRd",annot=True, xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Decision')
    plt.ylabel('True Class Label')
    positions = range(0,len(class_labels))
    #plt.xticks(positions, class_labels)
    #plt.yticks(positions, class_labels)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    plt.close()

def plot_correct_classified(data, subset=['x','y','z']):
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
    markers = ['v', '^', '<', '>', '8', 's', 'p', '*', 'h', '+', 'x', 'D']
    fig = plt.figure(figsize = (5.5,5))
    fig.subplots_adjust(left=0.01, right=0.96, top=0.99, bottom=0.01, wspace=0)
    ax = plt.axes(projection ="3d")
    # Plot correct
    correct = data[data['Correct']==True]
    true_labels = correct.sort_index().index.unique().tolist()
    print('Number of correct classified points: %d'%correct.shape[0])
    for i, true_label in enumerate(true_labels):
        temp = correct.loc[true_label, :]
        xs = temp[subset[0]].tolist()
        ys = temp[subset[1]].tolist()
        zs = temp[subset[2]].tolist()
        ax.scatter3D(xs, ys, zs, label=true_label, marker=markers[i], alpha=0.3, color='green')
    # Plot incorrect
    correct = data[data['Correct']==False]
    true_labels = correct.sort_index().index.unique().tolist()
    print('Number of incorrect classified points: %d'%correct.shape[0])
    for i, true_label in enumerate(true_labels):
        temp = correct.loc[true_label, :]
        xs = temp[subset[0]].tolist()
        ys = temp[subset[1]].tolist()
        zs = temp[subset[2]].tolist()
        ax.scatter3D(xs, ys, zs, label=true_label, marker=markers[i], alpha=0.3, color='red')
    ax.set_xlabel('%s'%subset[0])
    ax.set_ylabel('%s'%subset[1])
    ax.set_zlabel('%s'%subset[2])
    #ax.get_legend().remove()
    green_patch = mpatches.Patch(color='green', label='Correct')
    red_patch = mpatches.Patch(color='red', label='Incorrect')
    ax.legend(handles=[green_patch, red_patch], loc='upper left', title='Classification')
    plt.savefig('./%s_%s_%s_true_class_classified.pdf'%(subset[0], subset[1], subset[2]))
    plt.clf()
    return None

if __name__=='__main__':
    # Dimensions: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide,	total sulfur dioxide, density, pH, sulphates, alcohol
    wine_path = '/home/emily/Documents/intro_ml/homework_1/question_3/winequality-white.csv'
    wine_df = pd.read_csv(wine_path, delimiter=';', index_col='quality')
    print(wine_df)
    data_info = estimate_cov_mu(data=wine_df)
    print(data_info)
    # alcohol, pH, residual sugar
    #plot_subset(data=wine_df, subset=['alcohol', 'pH', 'residual sugar'])
    # citric acid, total sulfer dioxide, density
    #plot_subset(data=wine_df, subset=['citric acid', 'total sulfur dioxide', 'density'])
    wine_df = make_decisions(data=wine_df, data_info=data_info, loss_matrix=None, true_class_label='quality')
    print(wine_df)
    plot_correct_classified(data=wine_df, subset=['alcohol', 'pH', 'residual sugar'])
    plot_correct_classified(data=wine_df, subset=['citric acid', 'total sulfur dioxide', 'density'])
    plot_decision_matrix(data=wine_df, save_path='./wine_decision_matrix.pdf', true_class_label='quality')
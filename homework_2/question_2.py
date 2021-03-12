#from utility_functions import generateData as generate_data
import pandas as pd 
import numpy as np
import random
import matplotlib.pyplot as plt 
from numpy.random import default_rng
from scipy.stats import multivariate_normal
from math import floor, ceil

def generate_data(mus, sigmas, priors, N):
    '''
    Generate N samples according to a multivariate Gaussian probability density function, 
    keeping track of the true class labels for each sample. 

    Parameters
    ----------
    The mus, covs, and cps of the Gaussian distributions in array form

    Returns
    -------
    samples: numpy.array
        The generated sample data
    '''
    rng = default_rng()
    overall_size = N
    n = mus.shape[0]
    priors = np.cumsum(priors)
    size_1a = 0
    size_1b = 0
    size_2 = 0
    for i in range(0, overall_size) :
        r = random.random()
        if(r < priors[0]):
            size_1a = size_1a + 1
        elif(r < priors[1]):
            size_1b = size_1b + 1
        elif(r < priors[2]):
            size_2 = size_2 + 1

    samples_1a = rng.multivariate_normal(mean=mus[0], cov=sigmas[0], size=size_1a)
    samples_1a = pd.DataFrame(samples_1a, columns=['x','y'])
    samples_1a['True Class Label'] = 1

    samples_1b = rng.multivariate_normal(mean=mus[1], cov=sigmas[1], size=size_1b)
    samples_1b = pd.DataFrame(samples_1b, columns=['x','y'])
    samples_1b['True Class Label'] = 1

    samples_2 = rng.multivariate_normal(mean=mus[2], cov=sigmas[2], size=size_2)
    samples_2 = pd.DataFrame(samples_2, columns=['x','y'])
    samples_2['True Class Label'] = 2

    samples   = samples_1a.append([samples_1b, samples_2])
    return samples

def implement_classifier_and_plots(samples, mus, sigmas, priors, save_path='./ROC_curve.pdf'):
    '''
    Plots the minimum risk and ROC curve with theorectical and experimental probabilites.
    Parameters
    ----------
    samples_path: string
        File containing the sample data
    mus: array
        The vectors of mu for the two classes.
    sigmas: array
        The matrixes of sigma for the two classes.
    priors: array
        The probabilites of the labels.
    Returns
    -------
    exp_min: dict
        Info of the experimental minimum error.
    thy_min: dict
        Info of the theorectical minimum error.
    '''
    # Make decisions
    discriminants = []
    decisions = []
    prior_1 = (priors[0]+priors[1])
    prior_2 = priors[2]
    gamma = prior_1/prior_2
    print(gamma)
    w_1 = 1/2
    w_2 = 1/2
    for i in range(0, samples.shape[0]):
        sample = samples.iloc[i].to_numpy()[:-1]
        discriminant = (w_1*multivariate_normal.pdf(sample, mus[0], sigmas[0])+w_2*multivariate_normal.pdf(sample, mus[1], sigmas[1]))/multivariate_normal.pdf(sample, mus[2], sigmas[2])
        discriminants.append(discriminant)
        if(discriminant>gamma):
            decisions.append(1)
        else:
            decisions.append(2)
    samples['Discriminant'] = discriminants
    samples['Decision'] = decisions

    # Plot ROC curve
    samples = samples.sort_values('Discriminant')
    dis_0 = samples[samples['True Class Label']==1]['Discriminant'].tolist()
    dis_1 = samples[samples['True Class Label']==2]['Discriminant'].tolist()
    df = pd.DataFrame(columns=['False Positive', 'True Positive', 'Gamma', 'Probability Error'])
    for index, row in samples.iterrows():
        discriminant   = row['Discriminant'] 
        false_positive = len([class_dis for class_dis in dis_0 if class_dis>=discriminant])/len(dis_0)
        true_positive = len([class_dis for class_dis in dis_1 if class_dis>=discriminant])/len(dis_1)
        p_err = false_positive*prior_1+(1-true_positive)*prior_2
        d = {'False Positive': false_positive, 'True Positive': true_positive, 
             'Gamma': discriminant, 'Probability Error': p_err}
        df = df.append(d, ignore_index=True)
    df = df.sort_values('Probability Error')
    print(df)
    # Get info of minimum experimental probablility error
    exp_min = df.iloc[0]
    print('Experimental Mimimum Error Info:\n')
    print(exp_min)
    # Calculate theorectical error
    thy_gamma = gamma
    thy_lambdas = [len([class_dis for class_dis in dis_0 if class_dis>=thy_gamma])/len(dis_0),
                len([class_dis for class_dis in dis_1 if class_dis>=thy_gamma])/len(dis_1)]
    thy_p_err = thy_lambdas[0]*prior_1 + (1-thy_lambdas[1])*prior_2
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
    ax.legend(title='Minimum Error Probabilities', loc='upper left')
    #ax.set_title('Minimum Expected Risk ROC Curve')
    ax.set_xlabel('Probability of False Positive')
    ax.set_ylabel('Probability of True Positive')
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.xaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    plt.savefig('ROC_curve.pdf')
    plt.clf()
    plt.close()

    # Plot data set and outcomes
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    for idx,row in samples.iterrows():
        true_label = row['True Class Label']
        decision   = row['Decision']
        x = row['x']
        y = row['y']
        if(true_label==1):
            if(true_label==decision):
                ax.plot(x,y,'go', alpha=0.1)
            else:
                ax.plot(x,y,'ro', alpha=0.1)
        else:
            if(true_label==decision):
                ax.plot(x,y,'g^', alpha=0.1)
            else:
                ax.plot(x,y,'r^', alpha=0.1)
    plt.savefig('./q2_p1.pdf')

def mle(phi, t):
    # get pseudo-inverse
    tphi = np.transpose(phi)
    results = np.matmul(np.linalg.inv(np.matmul(tphi,phi)),tphi)
    # multiply by y
    results = np.matmul(results, t)
    return results

def mle_decisions(samples, ws_20, ws_200, ws_2000):
    # For 20 samples
    w_0 = ws_20[0,0]
    w_1 = ws_20[0,1]
    decisions = []
    for idx, row in samples.iterrows():
        x = row['x']
        y = row['y']
        prediction = w_0+w_1*x
        if(prediction<y):
            decisions.append(2)
        else:
            decisions.append(1)
    samples['Decision, 20'] = decisions
    # For 200 samples
    w_0 = ws_200[0,0]
    w_1 = ws_200[0,1]
    decisions = []
    for idx, row in samples.iterrows():
        x = row['x']
        y = row['y']
        prediction = w_0+w_1*x
        if(prediction<y):
            decisions.append(2)
        else:
            decisions.append(1)
    samples['Decision, 200'] = decisions
    # For 2000 samples
    w_0 = ws_2000[0,0]
    w_1 = ws_2000[0,1]
    decisions = []
    for idx, row in samples.iterrows():
        x = row['x']
        y = row['y']
        prediction = w_0+w_1*x
        if(prediction<y):
            decisions.append(2)
        else:
            decisions.append(1)
    samples['Decision, 2000'] = decisions
    return samples

def plot_classified_labels(samples, ws_20, ws_200, ws_2000):
    fig, axes = plt.subplots(1,3, sharey=True, sharex=True, figsize=(9,4))
    min_x = floor(samples['x'].min())
    max_x = ceil(samples['x'].max())
    x_span = np.linspace(min_x, max_x, num=1000)
    # Plot with 20 sample mle
    w_0 = ws_20[0,0]
    w_1 = ws_20[0,1]
    axes[0].set_xlim(min_x, max_x)
    incorrect = 0
    for idx, row in samples.iterrows():
        true_label = row['True Class Label']
        decision   = row['Decision, 20']
        x = row['x']
        y = row['y']
        if(true_label==1):
            if(true_label==decision):
                axes[0].plot(x,y,'go', alpha=0.1)
            else:
                axes[0].plot(x,y,'ro', alpha=0.1)
                incorrect = incorrect + 1
        else:
            if(true_label==decision):
                axes[0].plot(x,y,'g^', alpha=0.1)
            else:
                axes[0].plot(x,y,'r^', alpha=0.1)
                incorrect = incorrect + 1
    p_err_20 = incorrect/samples.shape[0]
    print(p_err_20)
    fx = []
    for i in range(len(x_span)):
        x = x_span[i]
        fx.append(w_0+w_1*x)
    fx = np.squeeze(fx)
    axes[0].plot(x_span,fx)
    # Plot with 200 sample mle
    w_0 = ws_200[0,0]
    w_1 = ws_200[0,1]
    incorrect = 0
    for idx, row in samples.iterrows():
        true_label = row['True Class Label']
        decision   = row['Decision, 200']
        x = row['x']
        y = row['y']
        if(true_label==1):
            if(true_label==decision):
                axes[1].plot(x,y,'go', alpha=0.1)
            else:
                axes[1].plot(x,y,'ro', alpha=0.1)
                incorrect = incorrect + 1
        else:
            if(true_label==decision):
                axes[1].plot(x,y,'g^', alpha=0.1)
            else:
                axes[1].plot(x,y,'r^', alpha=0.1)
                incorrect = incorrect + 1
    p_err_200 = incorrect/samples.shape[0]
    print(p_err_200)
    fx = []
    for i in range(len(x_span)):
        x = x_span[i]
        fx.append(w_0+w_1*x)
    fx = np.squeeze(fx)
    axes[1].plot(x_span,fx)
    # Plot with 2000 sample mle
    w_0 = ws_2000[0,0]
    w_1 = ws_2000[0,1]
    incorrect = 0
    for idx, row in samples.iterrows():
        true_label = row['True Class Label']
        decision   = row['Decision, 2000']
        x = row['x']
        y = row['y']
        if(true_label==1):
            if(true_label==decision):
                axes[2].plot(x,y,'go', alpha=0.1)
            else:
                axes[2].plot(x,y,'ro', alpha=0.1)
                incorrect = incorrect + 1
        else:
            if(true_label==decision):
                axes[2].plot(x,y,'g^', alpha=0.1)
            else:
                axes[2].plot(x,y,'r^', alpha=0.1)
                incorrect = incorrect + 1
    p_err_2000 = incorrect/samples.shape[0]
    print(p_err_2000)
    fx = []
    for i in range(len(x_span)):
        x = x_span[i]
        fx.append(w_0+w_1*x)
    fx = np.squeeze(fx)
    axes[2].plot(x_span,fx)
    fig.subplots_adjust(left=0.04, right=0.98, top=.89, bottom=0.10, wspace=0.05)
    fig.text(0.5, 0.01, 'X', va='center', ha='center')
    fig.text(0.01, 0.5, 'Y', va='center', ha='center', rotation=90)
    fig.text(0.5, 0.97, 'Training Data Set Size', va='center', ha='center')
    axes[0].seTheoretically Optimal Classifiert_title('N=20')
    axes[1].set_title('N=200')
    axes[2].set_title('N=2000')
    plt.savefig('./q2_p2.pdf')

if __name__=='__main__':
    priors = [.325,.325,.35]
    mus = np.array([[3, 0], [0, 3], [2, 2]])
    covs = np.zeros((3, 2, 2))
    covs[0,:,:] = np.array([[2, 0], [0, 1]])
    covs[1,:,:] = np.array([[1, 0], [0, 2]])
    covs[2,:,:] = np.array([[1, 0], [0, 1]])

    # Generate training data sets
    train_20 = generate_data(mus, covs, priors, 20)
    train_200 = generate_data(mus, covs, priors, 200)
    train_2000 = generate_data(mus, covs, priors, 2000)
    # Generate validation data set
    test = generate_data(mus, covs, priors, 10000)

    # Part 1
    #implement_classifier_and_plots(test, mus, covs, priors)

    # Part 2
    # Train with 20 samples
    phi = []
    N = len(train_20)
    for i in range(0,N,1):
        row = [1, train_20['x'].tolist()[i]]
        phi.append(row)
    phi = np.matrix(phi)
    t = train_20['y'].tolist()
    ws_20 = mle(phi, t) # gives the coefficients of linear regression
    # Train with 200 samples
    phi = []
    N = len(train_200)
    for i in range(0,N,1):
        row = [1, train_200['x'].tolist()[i]]
        phi.append(row)
    phi = np.matrix(phi)
    t = train_200['y'].tolist()
    ws_200 = mle(phi, t) 
    # Train with 200 samples
    phi = []
    N = len(train_2000)
    for i in range(0,N,1):
        row = [1, train_2000['x'].tolist()[i]]
        phi.append(row)
    phi = np.matrix(phi)
    t = train_2000['y'].tolist()
    ws_2000 = mle(phi, t)
    # Make decisions
    mle_decisions(test, ws_20, ws_200, ws_2000)
    # Plot
    plot_classified_labels(test, ws_20, ws_200, ws_2000)
from sklearn.neural_network import MLPClassifier
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
from sklearn.model_selection import cross_val_score, KFold
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler

def generate_data(N):
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
    # 4 classes
    # p = 0.25 for each class
    # 3d gaussian pdf
    priors = [0.25, 0.25, 0.25, 0.25]
    mu0 = [1,  1,  1]
    mu1 = [1, -1,  1]
    mu2 = [1, -1, -1]
    mu3 = [1,  1, -1]
    mus = [mu0,mu1,mu2,mu3]
    cov0 = [[1, 0.5, 0],
            [0, 1, 1],
            [1, 1, 0]]
    cov1 = [[0.5, 0,   1],
            [1.2, 1,   0.8],
            [1,   0.6, 0]]
    cov2 = [[1, 0.5, 1],
            [1, 1, 1],
            [1, 1, 1]]
    cov3 = [[0.5, 0.5, 0.5],
            [1.4, 1.2, 0.7],
            [0.6, 1.1, 0.5]]
    sigmas = [cov0,cov1,cov2,cov3]
    rng = default_rng()
    overall_size = N
    priors = np.cumsum(priors)
    size0 = 0
    size1 = 0
    size2 = 0
    size3 = 0
    for i in range(0, overall_size) :
        r = random.random()
        if(r < priors[0]):
            size0 = size0 + 1
        elif(r < priors[1]):
            size1 = size1 + 1
        elif(r < priors[2]):
            size2 = size2 + 1
        else:
            size3 = size3 + 1

    samples0 = rng.multivariate_normal(mean=mus[0], cov=sigmas[0], size=size0)
    samples0 = pd.DataFrame(samples0, columns=['x','y','z'])
    samples0['True Class Label'] = 0

    samples1 = rng.multivariate_normal(mean=mus[1], cov=sigmas[1], size=size1)
    samples1 = pd.DataFrame(samples1, columns=['x','y','z'])
    samples1['True Class Label'] = 1

    samples2 = rng.multivariate_normal(mean=mus[2], cov=sigmas[2], size=size2)
    samples2 = pd.DataFrame(samples2, columns=['x','y','z'])
    samples2['True Class Label'] = 2

    samples3 = rng.multivariate_normal(mean=mus[3], cov=sigmas[3], size=size3)
    samples3 = pd.DataFrame(samples3, columns=['x','y','z'])
    samples3['True Class Label'] = 3

    samples   = samples0.append([samples1, samples2, samples3])
    return samples

def plot_sample_data(samples):
    '''
    Plots the 3-dimensions of the samples taken from the distribution.

    Parameters
    ----------
    samples: pandas.DataFrame
        DataFrame containing the sample data

    Returns
    -------
    None
    '''
    fig = plt.figure(figsize=(5,5))
    fig.subplots_adjust(left=0.01, right=0.985, top=0.99, bottom=0.01, wspace=0)
    ax = plt.axes(projection ="3d")
    samples0 = samples[samples['True Class Label']==0]
    samples1 = samples[samples['True Class Label']==1]
    samples2 = samples[samples['True Class Label']==2]
    samples3 = samples[samples['True Class Label']==3]
    x_0 = samples0['x'].tolist()
    y_0 = samples0['y'].tolist()
    z_0 = samples0['z'].tolist()
    x_1 = samples1['x'].tolist()
    y_1 = samples1['y'].tolist()
    z_1 = samples1['z'].tolist()
    x_2 = samples2['x'].tolist()
    y_2 = samples2['y'].tolist()
    z_2 = samples2['z'].tolist()
    x_3 = samples3['x'].tolist()
    y_3 = samples3['y'].tolist()
    z_3 = samples3['z'].tolist()
    ax.scatter3D(x_0, y_0, z_0, label='1', marker='+', alpha=0.2)
    ax.scatter3D(x_1, y_1, z_1, label='2', marker='o', alpha=0.2)
    ax.scatter3D(x_2, y_2, z_2, label='3', marker='^', alpha=0.2)
    ax.scatter3D(x_3, y_3, z_3, label='4', marker='s', alpha=0.2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(loc='upper left', title='True Class Label')
    plt.savefig('./q1_original_generated_data.pdf')
    plt.clf()
    return None

def theorectically_optimal_classifier(samples):
    '''
    Using the knowledge of true data pdf, construct the minimum-probability-of-error 
    classification rule, apply it on the test dataset, and empirically estimate the 
    probability of error for this theoretically optimal classifier. This provides the 
    aspirational performance level for the MLP classfier.

    Parameters
    ----------
    samples: pandas.DataFrame
        DataFrame containing the sample data

    Returns
    -------
    p_err: float
        Theorectical minimum-probability-of-error for the classification.
    '''
    
    return None

def cross_validation(data, k=10, verbose=False):
    '''
    Cross-validation to determine best number of perceptrons
    for an MLPclassifier.

    Parameters
    ----------
    data: pd.DataFrame
        The data to use
    k: int, optional
        Number of folds; k-fold cross validation

    Returns
    -------
    P: int
        Optimal number of perceptrons.
    results: pandas.DataFrame
        Dataframe showing the probability of error and number
        of perceptrons.
    '''
    # Make k subsamples
    cv = KFold(n_splits=k)
    # Repeat the cross-validation over numerous hyparameter configurations
    Ps = [x for x in np.arange(1, 11, 1)]
    activation = 'relu'
    results = pd.DataFrame()
    X = []
    for i, row in data.iterrows():
        X.append([row['x'],row['y'],row['z']])
    for p in Ps:
        model = MLPClassifier(hidden_layer_sizes=(p,), activation=activation, max_iter=4000)
        scores = cross_val_score(estimator=model, X=X, y=data['True Class Label'], cv=cv, scoring='accuracy')
        print(scores)
        p_errs = [1-score for score in scores]
        d = {'Mean Probability of Error Score': np.mean(p_errs), 'Number of Perceptrons': p}
        results = results.append(d, ignore_index=True)
    data = results
    results = results.sort_values(by='Mean Probability of Error Score').iloc[0]
    min_score = results['Mean Probability of Error Score']
    if(verbose):
        print('The minimum-average-cross-validation-probability-of-error is: %.3f'%min_score)
    return results['Number of Perceptrons'], data

def plot_cross_validation_results(data):
    '''
    Plot how the hyperparameter changes the cross validation results.

    Parameters
    ----------
    Data: pd.DataFrame
        Data on the cross validation previously done.
    
    Returns
    -------
    None
    '''
    fig, ax = plt.subplots(1,1,figsize=(5,2.8))
    fig.subplots_adjust(left=0.11, right=0.985, top=0.98, bottom=0.16, wspace=0)
    data['Training Dataset Size'] = data['Training Dataset Size'].astype(str)
    sns.scatterplot(data=data, x='Number of Perceptrons', y='Mean Probability of Error Score', hue='Training Dataset Size', palette='ch:s=-.2,r=.6')
    ax.set_ylim(0,1)
    plt.savefig('./q1_cross_validation.pdf')
    plt.clf()
    plt.close()
    return None

def test_hyperparameters(train_data, test_data, p):
    '''
    Trains an MLP with the appropriate number of perceptrons, determined previously by
    10-fold cross validation, using the entire respective training set.

    Parameters
    ----------
    train_data: pd.DataFrame
        Data to train the model
    test_data: pd.DataFrame
        Data to test the accuracy of the model
    p: int
        Number of perceptrons to use in the model.

    Returns
    -------
    test_data: pd.DataFrame
        Predicted values and original testing data.
    '''
    # Train the model using the optimized hyperparameters
    X = []
    for i, row in train_data.iterrows():
        X.append([row['x'],row['y'],row['z']])
    Y = train_data['True Class Label']
    activation = 'relu'
    model = MLPClassifier(hidden_layer_sizes=(int(p),), activation=activation, max_iter=4000)
    model.fit(X,Y)
    # Test the accuracy of the model
    X = []
    for i, row in test_data.iterrows():
        X.append([row['x'],row['y'],row['z']])
    Y = model.predict(X)
    test_data['Predicted Label'] = Y
    return test_data

def plot_model_results(data, training_size):
    '''
    Plot the accuracy of determined hyperparameters in training the model.

    Parameters
    ----------
    Data: pd.DataFrame
        Data on the model fit previously done.
    training_size: int
        Number of sample in training dataset.
    
    Returns
    -------
    None
    '''
    fig = plt.figure(figsize=(5,5))
    fig.subplots_adjust(left=0.01, right=0.985, top=0.99, bottom=0.01, wspace=0)
    ax = plt.axes(projection ="3d")
    correct = 0
    for idx,row in data.iterrows():
        if idx%10 != 0:
            continue
        true_label = row['True Class Label']
        decision   = row['Predicted Label']
        x = row['x']
        y = row['y']
        z = row['z']
        if(true_label==0.0):
            if(true_label==decision):
                ax.scatter3D(x,y,z,marker='x',color='g',alpha=0.1)
                correct = correct + 1
            else:
                ax.scatter3D(x,y,z,marker='x',color='r',alpha=0.1)
        elif(true_label==1.0):
            if(true_label==decision):
                ax.scatter3D(x,y,z,marker='o',color='g',alpha=0.1)
                correct = correct + 1
            else:
                ax.scatter3D(x,y,z,marker='o',color='r',alpha=0.1)
        elif(true_label==2.0):
            if(true_label==decision):
                ax.scatter3D(x,y,z,marker='^',color='g',alpha=0.1)
                correct = correct + 1
            else:
                ax.scatter3D(x,y,z,marker='^',color='r',alpha=0.1)
        else:
            if(true_label==decision):
                ax.scatter3D(x,y,z,marker='s',color='g',alpha=0.1)
                correct = correct + 1
            else:
                ax.scatter3D(x,y,z,marker='s',color='r',alpha=0.1)
    accuracy = correct/(data.shape[0]/10)
    print('Model accuracy was %.3f'%(accuracy))
    '''
    legend_elements = [Line2D([0], [0], marker='x', color='w', label='1', markerfacecolor='grey', markersize=15),
                        Line2D([0], [0], marker='o', color='w', label='2', markerfacecolor='grey', markersize=15),
                        Line2D([0], [0], marker='^', color='w', label='3', markerfacecolor='grey', markersize=15),
                        Line2D([0], [0], marker='s', color='w', label='4', markerfacecolor='grey', markersize=15)]
    legend0 = ax.legend(handles=legend_elements, title='True Class Label', loc='upper right')
    '''
    legend_elements = [Patch(facecolor='g', edgecolor='g', label='True'),
                        Patch(facecolor='r', edgecolor='r', label='False')]
    legend1 = ax.legend(handles=legend_elements, title='Correctly Classified', loc='lower right')
    #ax.add_artist(legend0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('./q1_classified_data_%d.jpg'%training_size)
    plt.clf()
    return accuracy

def plot_accuracy(accuracy_data, test_data):
    '''
    Scatterplot of how the size of the training dataset affects the accuracy
    of the model.

    Parameters
    ----------
    data: pd.DataFrame
        Data for plotting

    Returns
    -------
    None
    '''
    # Get the theoretical optimal classification
    thy_optimal = find_optimal_classification(test_data)
    print("Theoretical optimal classification: %.3f"%thy_optimal)
    fig, ax = plt.subplots(1,1,figsize=(5,2))
    fig.subplots_adjust(left=0.13, right=0.985, top=0.97, bottom=0.23, wspace=0)
    data['Training Dataset Size'] = data['Training Dataset Size'].astype(int).astype(str)
    sns.scatterplot(data=data, x='Training Dataset Size', y='Accuracy', palette='ch:s=-.2,r=.6')
    plt.axhline(y=thy_optimal, color='r', linestyle='--')
    ax.set_ylim(.7,.9)
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.savefig('./q1_model_accuracy.pdf')
    plt.clf()
    plt.close()
    return None

def find_optimal_classification(data):
    X = []
    for i, row in data.iterrows():
        X.append([row['x'],row['y'],row['z']])
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    Y = data['True Class Label']
    model = GaussianNB(priors=[0.25,0.25,0.25,0.25])
    model.fit(X,Y)
    data['Predicted Label'] = model.predict(X)
    correct = 0
    for idx,row in data.iterrows():
        true_label = row['True Class Label']
        decision   = row['Predicted Label']
        if(true_label==decision):
                correct = correct + 1
    accuracy = correct/(data.shape[0])
    return accuracy

if __name__=='__main__':
    # Generate samples
    # Training datasets
    train0 = generate_data(100)
    train1 = generate_data(200)
    train2 = generate_data(500)
    train3 = generate_data(1000)
    train4 = generate_data(2000)
    train5 = generate_data(5000)
    # Testing dataset
    test = generate_data(100000)

    # Plot the testing data without classification
    plot_sample_data(test)
    # Find optimal hyperparameters
    final_results = pd.DataFrame()
    n_perceptrons, results = cross_validation(train0, k=10, verbose=False)
    print("100 CV Perceptrons: %d"%n_perceptrons)
    results.to_csv('./q1_cv_results_100.csv')
    results["Training Dataset Size"] = 100
    final_results = final_results.append(results, ignore_index=True)

    n_perceptrons, results = cross_validation(train1, k=10, verbose=False)
    print("200 CV Perceptrons: %d"%n_perceptrons)
    results.to_csv('./q1_cv_results_200.csv')
    results["Training Dataset Size"] = 200
    final_results = final_results.append(results, ignore_index=True)

    n_perceptrons, results = cross_validation(train2, k=10, verbose=False)
    print("500 CV Perceptrons: %d"%n_perceptrons)
    results.to_csv('./q1_cv_results_500.csv')
    results["Training Dataset Size"] = 500
    final_results = final_results.append(results, ignore_index=True)

    n_perceptrons, results = cross_validation(train3, k=10, verbose=False)
    print("1000 CV Perceptrons: %d"%n_perceptrons)
    results.to_csv('./q1_cv_results_1000.csv')
    results["Training Dataset Size"] = 1000
    final_results = final_results.append(results, ignore_index=True)

    n_perceptrons, results = cross_validation(train4, k=10, verbose=False)
    print("2000 CV Perceptrons: %d"%n_perceptrons)
    results.to_csv('./q1_cv_results_2000.csv')
    results["Training Dataset Size"] = 2000
    final_results = final_results.append(results, ignore_index=True)

    n_perceptrons, results = cross_validation(train5, k=10, verbose=False)
    print("5000 CV Perceptrons: %d"%n_perceptrons)
    results.to_csv('./q1_cv_results_5000.csv')
    results["Training Dataset Size"] = 5000
    final_results = final_results.append(results, ignore_index=True)

    final_results.to_csv('./q1_cv_results_overall.csv')
    # Plot how accuracy changes with perceptron hyperparameter changes
    data = pd.read_csv('./q1_cv_results_overall.csv', index_col=0)
    plot_cross_validation_results(data)

    # Choose minimal error number of perceptrons
    final_results = pd.DataFrame()

    training_size = 100
    df = pd.read_csv('./q1_cv_results_%d.csv'%training_size, index_col=0)
    df = df.sort_values(by='Mean Probability of Error Score').iloc[0]
    p = df['Number of Perceptrons']
    results = test_hyperparameters(train0, test, p)
    accuracy = plot_model_results(results, training_size)
    final_results = final_results.append({'Accuracy': accuracy, 'Training Dataset Size': training_size}, ignore_index=True)

    training_size = 200
    df = pd.read_csv('./q1_cv_results_%d.csv'%training_size, index_col=0)
    df = df.sort_values(by='Mean Probability of Error Score').iloc[0]
    p = df['Number of Perceptrons']
    results = test_hyperparameters(train1, test, p)
    accuracy = plot_model_results(results, training_size)
    final_results = final_results.append({'Accuracy': accuracy, 'Training Dataset Size': training_size}, ignore_index=True)

    training_size = 500
    df = pd.read_csv('./q1_cv_results_%d.csv'%training_size, index_col=0)
    df = df.sort_values(by='Mean Probability of Error Score').iloc[0]
    p = df['Number of Perceptrons']
    results = test_hyperparameters(train2, test, p)
    accuracy = plot_model_results(results, training_size)
    final_results = final_results.append({'Accuracy': accuracy, 'Training Dataset Size': training_size}, ignore_index=True)

    training_size = 1000
    df = pd.read_csv('./q1_cv_results_%d.csv'%training_size, index_col=0)
    df = df.sort_values(by='Mean Probability of Error Score').iloc[0]
    p = df['Number of Perceptrons']
    results = test_hyperparameters(train3, test, p)
    accuracy = plot_model_results(results, training_size)
    final_results = final_results.append({'Accuracy': accuracy, 'Training Dataset Size': training_size}, ignore_index=True)

    training_size = 2000
    df = pd.read_csv('./q1_cv_results_%d.csv'%training_size, index_col=0)
    df = df.sort_values(by='Mean Probability of Error Score').iloc[0]
    p = df['Number of Perceptrons']
    results = test_hyperparameters(train4, test, p)
    accuracy = plot_model_results(results, training_size)
    final_results = final_results.append({'Accuracy': accuracy, 'Training Dataset Size': training_size}, ignore_index=True)

    training_size = 5000
    df = pd.read_csv('./q1_cv_results_%d.csv'%training_size, index_col=0)
    df = df.sort_values(by='Mean Probability of Error Score').iloc[0]
    p = df['Number of Perceptrons']
    results = test_hyperparameters(train5, test, p)
    accuracy = plot_model_results(results, training_size)
    final_results = final_results.append({'Accuracy': accuracy, 'Training Dataset Size': training_size}, ignore_index=True)

    final_results.to_csv('./q1_model_accuracy.csv')
    
    data = pd.read_csv('./q1_model_accuracy.csv', index_col=0)
    plot_accuracy(data,test)
from generateMultiringDataset import generateMultiringDataset as generate_data
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def cross_validation(data, k=10, verbose=False):
    '''
    Select the best box constraint hyperparameter C and the Gaussian kernel width 
    parameter σ.  Use minimum-average-cross-validation-probability-of-error to 
    select best hyperparameters.

    Parameters
    ----------
    data: pd.DataFrame
        The data to use
    k: int, optional
        Number of folds; k-fold cross validation

    Returns
    -------
    C: float
        The C hyperparameter.
    Sigma: float
        The Gaussian kernel width parameter.
    '''
    # Make k subsamples
    cv = KFold(n_splits=k)
    # Repeat the cross-validation over numerous hyparameter configurations
    C = np.arange(0.5,10.0,0.5)
    sigmas = np.arange(0.25,10.0,0.25)
    gammas = [1/sigma for sigma in sigmas] # sigma = 1/gamma
    results = pd.DataFrame()
    X = []
    for i, row in data.iterrows():
        X.append([row['x'],row['y']])
    for c in C:
        for gamma in gammas:
            model = SVC(C=c, kernel='rbf', gamma=gamma)
            scores = cross_val_score(estimator=model, X=X, y=data['Label'], cv=cv, scoring='accuracy')
            p_errs = [1-score for score in scores]
            d = {'Mean Probability of Error Score': np.mean(p_errs), 'C': c, 'Sigma': 1/gamma, 'Gamma': gamma}
            results = results.append(d, ignore_index=True)
    data = results
    results = results.sort_values(by='Mean Probability of Error Score').iloc[0]
    min_score = results['Mean Probability of Error Score']
    if(verbose):
        print('The minimum-average-cross-validation-probability-of-error is: %.3f'%min_score)
    return results['C'], results['Gamma'], data

def test_hyperparameters(train_data, test_data, C, gamma):
    '''
    Train a final SVM using the best combination of hyperparameters with the entire training set. 
    Classify the testing dataset samples with this trained SVM to assess performance; estimate 
    the probability error using the test set. 

    Parameters
    ----------
    train_data: pd.DataFrame
        Data to train the model
    test_data: pd.DataFrame
        Data to test the accuracy of the model
    C: float
        The C hyperparameter.
    Sigma: float
        The Gaussian kernel width parameter.

    Returns
    -------
    test_data: pd.DataFrame
        Predicted values and original testing data.
    '''
    # Train the model using the optimized hyperparameters
    X = []
    for i, row in train_data.iterrows():
        X.append([row['x'],row['y']])
    Y = train_data['Label']
    model = SVC(C=C, kernel='rbf', gamma=gamma)
    model.fit(X,Y)
    # Test the accuracy of the model
    X = []
    for i, row in test_data.iterrows():
        X.append([row['x'],row['y']])
    Y = model.predict(X)
    test_data['Predicted Label'] = Y
    return test_data

def cross_validation_results(data):
    '''
    Plot how the hyperparameters change the cross validation results.

    Parameters
    ----------
    Data: pd.DataFrame
        Data on the cross validation previously done.
    
    Returns
    -------
    None
    '''
    data['Accuracy'] = 1-data['Mean Probability of Error Score']
    fig, axes = plt.subplots(1,2,sharey=True,figsize=[5,2])
    sns.regplot(data=data,x='Sigma',y='Accuracy',color='skyblue',ax=axes[0],ci=0,truncate=False,order=3,line_kws={'color':'grey'})
    sns.regplot(data=data,x='C',y='Accuracy',color='maroon',ax=axes[1],ci=0,truncate=False,order=3,line_kws={'color':'grey'})
    axes[1].set_ylabel('')
    vals = axes[0].get_yticks()
    axes[0].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    axes[0].set_xlim(0,10)
    axes[1].set_xlim(0,10)
    axes[0].set_ylim(.90,.95)
    axes[1].set_ylim(.90,.95)
    axes[0].set_xticks(np.arange(0,12,2))
    axes[0].set_xticklabels(np.arange(0,12,2))
    axes[1].set_xticks(np.arange(0,12,2))
    axes[1].set_xticklabels(np.arange(0,12,2))
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].xaxis.grid(color='lightgrey', linestyle=':')
    axes[1].set_axisbelow(True)
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].xaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    fig.subplots_adjust(left=0.12, right=0.965, top=.94, bottom=0.22, wspace=0.10, hspace=0.18)
    plt.savefig('./q2_cross_validation.pdf')
    plt.clf()
    plt.close()
    return None

def plot_model_results(data):
    '''
    Plot the accuracy of determined hyperparameters in training the model.

    Parameters
    ----------
    Data: pd.DataFrame
        Data on the model fit previously done.
    
    Returns
    -------
    None
    '''
    fig, ax = plt.subplots(1, 1, figsize=[5,4.6])
    correct = 0
    for idx,row in data.iterrows():
        true_label = row['Label']
        decision   = row['Predicted Label']
        x = row['x']
        y = row['y']
        if(true_label==1.0):
            if(true_label==decision):
                ax.plot(x,y,'go', alpha=0.1)
                correct = correct + 1
            else:
                ax.plot(x,y,'ro', alpha=0.1)
        else:
            if(true_label==decision):
                ax.plot(x,y,'g^', alpha=0.1)
                correct = correct + 1
            else:
                ax.plot(x,y,'r^', alpha=0.1)
    print('Model accuracy was %.3f'%(correct/data.shape[0]))
    ax.set_ylim(-30,30)
    ax.set_xlim(-30,30)
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.xaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='1', markerfacecolor='grey', markersize=15),
                        Line2D([0], [0], marker='^', color='w', label='0', markerfacecolor='grey', markersize=15)]
    legend0 = ax.legend(handles=legend_elements, title='True Class Label', loc='upper right')
    legend_elements = [Patch(facecolor='g', edgecolor='g', label='True'),
                        Patch(facecolor='r', edgecolor='r', label='False')]
    legend1 = ax.legend(handles=legend_elements, title='Correctly Classified', loc='lower right')
    ax.add_artist(legend0)
    fig.subplots_adjust(left=0.12, right=0.965, top=.96, bottom=0.10)
    plt.savefig('./q2_model_results.pdf')
    plt.clf()
    plt.close()
    return None

if __name__=='__main__':
    # Generate data and convert to pandas.DataFrame
    n_labels = 2
    N = 1000
    data, labels = generate_data(n_labels, N)
    train = pd.DataFrame({'x': data[0], 'y': data[1], 'Label': labels[0]})
    N = 10000
    data, labels = generate_data(n_labels, N)
    test = pd.DataFrame({'x': data[0], 'y': data[1], 'Label': labels[0]})
    # Run functions
    '''
    C, gamma, data = cross_validation(train, verbose=True)
    data.to_csv('./q2_cross_validation_results.csv')
    print('C: %.3f'%C)
    print('Sigma: %.3f'%(1/gamma))
    results = test_hyperparameters(train, test, C, gamma)
    results.to_csv('./q2_model_results.csv')
    '''
    # Plot cross validation results
    '''
    data = pd.read_csv('./q2_cross_validation_results.csv', index_col=0)
    cross_validation_results(data)
    '''
    # Plot model results
    data = pd.read_csv('./q2_model_results.csv', index_col=0)
    plot_model_results(data)
from generate_data import hw2q1 as generate_data
import pandas as pd 
from sklearn.neural_network import MLPClassifier
import numpy as np 
from numpy.random import default_rng
import matplotlib.pyplot as plt 
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
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.losses import MeanSquaredError

def cross_validation(train_data, k=10):
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
    global PERCEPTRONS
    X = np.array([train_data['x0'], train_data['x1']]).reshape((1000,2)) #.T.tolist()
    #Y = np.asarray(train_data['y'], dtype="|S6")
    Y = train_data['y']
    Ps = [x for x in np.arange(1, 11, 1)]
    cv = KFold(n_splits=k)
    results = pd.DataFrame()
    for p in Ps:
        PERCEPTRONS = p
        #errors = []
        #for fold in folds:
        #model.fit(X_train,y_train, epochs = 100,verbose=0)  
        model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
        print("starting")
        scores = cross_val_score(estimator=model, X=X, y=Y, cv=cv, scoring='neg_mean_squared_error', error_score='raise')
        print("done")
        d = {'Mean Squared Error': np.absolute(np.mean(scores)), 'Number of Perceptrons': p}
        results = results.append(d, ignore_index=True)
    return results

def create_model():
    model = Sequential()
    model.add(Dense(units=PERCEPTRONS, activation='softplus', input_dim=2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model

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
    X = np.array([train_data['x0'], train_data['x1']]).reshape((1000,2)) #.T.tolist()
    Y = train_data['y']
    model = Sequential()
    model.add(Dense(units=p, activation='softplus', input_dim=2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    model.fit(X, Y)
    # Test the accuracy of the model
    X = np.array([test_data['x0'], test_data['x1']]).reshape((10000,2)) #.T.tolist()
    Y = model.predict(X)
    test_data['Predicted Label'] = Y
    return test_data

def plot_model_results(data):
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
        true_label = row['y']
        decision   = row['Predicted Label']
        x = row['x0']
        y = row['x1']
        ax.scatter3D(x,y,true_label, marker='.',color='g',alpha=0.1)
        ax.scatter3D(x,y,decision, marker='.',color='b',alpha=0.1)
    mse = MeanSquaredError()
    accuracy = mse(data['y'], data['Predicted Label']).numpy()
    print('Model accuracy was %.3f'%(accuracy))
    legend_elements = [Patch(facecolor='g', edgecolor='g', label='Original'),
                        Patch(facecolor='b', edgecolor='b', label='Predicted')]
    legend1 = ax.legend(handles=legend_elements, title='Value', loc='upper right')
    #ax.add_artist(legend0)
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('y')
    plt.savefig('./q1_classified_data.jpg')
    plt.clf()
    return accuracy

if __name__=='__main__':
    # train has 1000 samples, validate has 10000 samples
    x_train, y_train, x_test, y_test = generate_data()
    train = pd.DataFrame({'x0':x_train[0],'x1':x_train[1],'y':y_train})
    test = pd.DataFrame({'x0':x_test[0],'x1':x_test[1],'y':y_test})
    #print(test['y'].unique().tolist())
    '''
    phi = []
    for i, row in train.iterrows():
        r = [1, row['x0'], row['x1'], row['x0']**2, 
                row['x1']**2, row['x0']**3, row['x1']**3]
        phi.append(r)
    phi = np.matrix(phi)
    '''
    '''
    results = cross_validation(train, k=10)
    results.to_csv('./q1_cv_results.csv')
    '''
    #df = pd.read_csv('./q1_cv_results.csv', index_col=0)
    #df = df.sort_values(by='Mean Squared Error').iloc[0]
    #p = df['Number of Perceptrons']
    results = test_hyperparameters(train, test, 10)
    results.to_csv('q1_classified_results.csv')
    accuracy = plot_model_results(results)
    print(accuracy)
    
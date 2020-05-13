import numpy as np
import pandas as pd
import sklearn
import scipy.special
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn import metrics
import dynesty

def categorical_accuracy(y_true, y_pred):
    correct = 0
    total = 0
    for i in range(len(y_true)):
        act_label = np.argmax(y_true[i]) 
        pred_label = np.argmax(y_pred[i]) 
        if(act_label == pred_label):
            correct += 1
        total += 1
    accuracy = (correct/total)
    return accuracy

def sigmoid(x):
    return scipy.special.expit(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return (e_x.T / e_x.sum(axis=1)).T 
 
def multi_cross_entropy(predictions, targets):   
    ce = -np.sum(targets*np.log(predictions+1e-9)) # No division by N
    return ce

def single_cross_entropy(predictions, targets):
    N = predictions.shape[0]
    ce = sklearn.metrics.log_loss(targets, predictions)
    return ce * N

def mean_squared_error(predictions, targets):
    mse = metrics.mean_squared_error(predictions, targets)
    return mse

def regression_neural_network(X, w_1, b_1, w_2, b_2):
    hidden_layer = np.tanh((X @ w_1) + b_1.T)
    y = hidden_layer @ w_2 + b_2.T# no bias required on this layer
    return y

def multi_class_neural_network(X, w_1, b_1, w_2, b_2):
    hidden_layer = np.tanh((X @ w_1) + b_1.T)
    y = softmax((hidden_layer @ w_2) + b_2.T)
    return y

def single_class_neural_network(X, w_1, b_1, w_2, b_2):
    hidden_layer = np.tanh((X @ w_1) + b_1.T)
    y = sigmoid((hidden_layer @ w_2) + b_2.T)
    return y

def regression_predictions(X, W, input_neurons, hidden_neurons, output_neurons):
    param_1 = input_neurons * hidden_neurons
    bias_1  = hidden_neurons
    w_1 = W[0:param_1]
    w_1 = w_1.reshape((input_neurons,hidden_neurons))
    b_1 = W[param_1:param_1+ bias_1]
    
    param_2 = hidden_neurons * output_neurons
    bias_2 = output_neurons
    w_2 = W[param_1 + bias_1:param_1 + bias_1 + param_2]
    w_2 = w_2.reshape((hidden_neurons, output_neurons))
    b_2 = W[param_1 + bias_1 + param_2:param_1+ 
            bias_1 + param_2 + bias_2]

    predictions = regression_neural_network(X, w_1, b_1, w_2,b_2)
    
    return predictions


def multi_class_predictions(X, W, input_neurons, hidden_neurons, output_neurons): 
    param_1 = input_neurons * hidden_neurons
    bias_1  =  hidden_neurons
    w_1 = W[0:param_1]
    w_1 = w_1.reshape((input_neurons,hidden_neurons))
    b_1 = W[param_1:param_1+ bias_1]
    
    param_2 = hidden_neurons * output_neurons
    bias_2 = output_neurons
    w_2 = W[param_1 + bias_1:param_1 + bias_1 + param_2]
    w_2 = w_2.reshape((hidden_neurons, output_neurons))
    b_2 = W[param_1 + bias_1 + param_2:param_1+ 
            bias_1 + param_2 + bias_2]

    predictions = multi_class_neural_network(X, w_1, b_1, w_2, b_2)
    
    return predictions

def single_class_predictions(X, W, input_neurons, hidden_neurons, output_neurons): 
    param_1 = input_neurons * hidden_neurons
    bias_1  =  hidden_neurons
    w_1 = W[0:param_1]
    w_1 = w_1.reshape((input_neurons,hidden_neurons))
    b_1 = W[param_1:param_1+ bias_1]
    
    param_2 = hidden_neurons * output_neurons
    bias_2 = output_neurons
    w_2 = W[param_1 + bias_1:param_1 + bias_1 + param_2]
    w_2 = w_2.reshape((hidden_neurons, output_neurons))
    b_2 = W[param_1 + bias_1 + param_2:param_1+ 
            bias_1 + param_2 + bias_2]

    predictions = single_class_neural_network(X, w_1, b_1, w_2, b_2)
    
    return predictions


def return_results_regression(x, y, results_list, x_axis,
                              input_neurons, output_neurons):
    logZ =[]
    for i in results_list:
        logZ.append(i.logz[-1])

    samples_list = []
    weights_list = []
    index_max_list = []
    predictions_list_mode = []
    
    for i in results_list:
        samples, weights = i.samples, np.exp(i.logwt - i.logz[-1])
        samples_list.append(samples)
        weights_list.append(weights)
        index_max_list.append(np.argmax(weights))
 
    for i  in x_axis: 
        hidden_neurons = i
        samples = samples_list[i-1]
        weights = weights_list[i-1]
        W_mode = samples[index_max_list[i-1]]
        predictions_list_mode.append(
            regression_predictions(x, W_mode, input_neurons,hidden_neurons, output_neurons))    
    mse_list_mode = []
    for i  in x_axis:
        y_pred_mode = predictions_list_mode[i-1]
        mse_list_mode.append(metrics.mean_squared_error(y, y_pred_mode))
        
    return logZ, mse_list_mode


def return_results_regression_simple(results_list):
    logZ =[]
    for i in results_list:
        logZ.append(i.logz[-1])
    return logZ

def return_results_regression_mse(x, y, results_list, x_axis,
                              input_neurons, output_neurons):

    samples_list = []
    weights_list = []
    index_max_list = []
    predictions_list_mode = []
    
    for i in results_list:
        samples, weights = i.samples, np.exp(i.logwt - i.logz[-1])
        samples_list.append(samples)
        weights_list.append(weights)
        index_max_list.append(np.argmax(weights))
 
    for i  in x_axis: 
        hidden_neurons = i
        samples = samples_list[i-1]
        weights = weights_list[i-1]
        W_mode = samples[index_max_list[i-1]]
        predictions_list_mode.append(
            regression_predictions(x, W_mode, input_neurons,hidden_neurons, output_neurons))    
    mse_list_mode = []
    for i  in x_axis:
        y_pred_mode = predictions_list_mode[i-1]
        mse_list_mode.append(metrics.mean_squared_error(y, y_pred_mode))
        
    return mse_list_mode


def multi_class_return_results(x, y, results_list, x_axis, 
                         input_neurons, output_neurons):
    logZ =[]
    for i in results_list:
        logZ.append(i.logz[-1])

    samples_list = []
    weights_list = []
    index_max_list = []
    predictions_list = []
    
    for i in results_list:
        samples, weights = i.samples, np.exp(i.logwt - i.logz[-1])
        samples_list.append(samples)
        weights_list.append(weights)
        index_max_list.append(np.argmax(weights))
 
    for i  in x_axis: 
        hidden_neurons = i
        samples = samples_list[i-1]
        weights = weights_list[i-1]
        #W, cov = dynesty.utils.mean_and_cov(samples, weights)
        W = samples[index_max_list[i-1]]
        predictions_list.append(multi_class_predictions(x, W, 
                                            input_neurons, hidden_neurons, output_neurons))
    
    accuracy_list = []
    for i  in x_axis:
        y_pred = predictions_list[i-1]
        accuracy_list.append(categorical_accuracy(y, y_pred))

    return logZ, accuracy_list

def single_class_return_results(x, y, results_list, x_axis, 
                         input_neurons, output_neurons):
    logZ =[]
    for i in results_list:
        logZ.append(i.logz[-1])

    samples_list = []
    weights_list = []
    index_max_list = []
    predictions_list = []
    
    for i in results_list:
        samples, weights = i.samples, np.exp(i.logwt - i.logz[-1])
        samples_list.append(samples)
        weights_list.append(weights)
        index_max_list.append(np.argmax(weights))
 
    for i  in x_axis: 
        hidden_neurons = i
        samples = samples_list[i-1]
        weights = weights_list[i-1]
        W_mean, cov = dynesty.utils.mean_and_cov(samples, weights)
        W_mode = samples[index_max_list[i-1]]
        predictions_list.append(np.round(single_class_predictions(x, W_mode, 
                                            input_neurons, hidden_neurons, output_neurons)))
    
    accuracy_list = []
    for i  in x_axis:
        y_pred = predictions_list[i-1]
        accuracy_list.append(metrics.accuracy_score(y, y_pred))

    return logZ, accuracy_list, predictions_list


''' Loading in the data'''

def return_iris_processed_data():
    iris = datasets.load_iris()
    
    X = iris['data']
    y = iris['target']
    
    # One hot encoding of the output variable
    enc = OneHotEncoder()
    Y = enc.fit_transform(y[:, np.newaxis]).toarray()
    
    # Scale features to have mean 0 and variance 1 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data set into training and testing
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size = 0.3, random_state = 1)
    
    return x_train, y_train, x_test, y_test

def return_boston_processed_data():
    X, Y = datasets.load_boston(return_X_y =True)
        
    # Scale features to have mean 0 and variance 1 
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data set into training and testing
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size = 0.3, random_state = 1)
    
    return x_train, y_train, x_test, y_test

def return_protein_processed_data():
    dataset = pd.read_csv("Protein_Data.csv", header = None) 
     #Split the data into training and test set
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values
    
    # Scale features to have mean 0 and variance 1 
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.3,random_state = 1)
    
    return x_train, y_train, x_test, y_test

def return_yacht_processed_data():
    dataset = pd.read_csv("Yacht_Data.txt", sep=" ", header=None) 
     #Split the data into training and test set
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values
    
    # Scale features to have mean 0 and variance 1 
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.3,random_state = 1)
    
    return x_train, y_train, x_test, y_test


def return_diabetes_processed_data():
    X, Y = datasets.load_diabetes(return_X_y =True)
        
    # Scale features to have mean 0 and variance 1 
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data set into training and testing
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size = 0.3, random_state = 1)
    
    return x_train, y_train, x_test, y_test

def return_concrete_processed_data():
    dataset = pd.read_excel("Concrete_Data.xls")
        
    #Split the data into training and test set
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values
    
    # Scale features to have mean 0 and variance 1 
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.3,random_state = 1)
    
    return x_train, y_train, x_test, y_test

def return_power_processed_data():
    dataset = pd.read_excel("Power_Data.xlsx")
        
    #Split the data into training and test set
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values
    
    # Scale features to have mean 0 and variance 1 
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.3,random_state = 1)
    
    return x_train, y_train, x_test, y_test

def return_mnist_processed_data():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    x_train = x_train.reshape(60000, 784)
    x_test =  x_test.reshape(10000, 784)
    x_train =x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    # convert class vectors to binary class matrices
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    y_train = np.reshape(y_train, (60000,num_classes))
    y_test = np.reshape(y_test, (10000,num_classes))
    
    return x_train, y_train, x_test, y_test

def return_taiwan_processed_data():
    #Load the data
    dataset = pd.read_excel("default_data.xls")
    dataset.index = dataset['ID']
    dataset.drop('ID',axis=1,inplace=True)
    dataset['SEX'].value_counts(dropna=False)
    dataset['EDUCATION'].value_counts(dropna=False)
    dataset = dataset.rename(columns={'PAY_0': 'PAY_1'})
    
    # Clean the data
    fil = (dataset.EDUCATION == 5) | (dataset.EDUCATION == 6) | (dataset.EDUCATION == 0)
    dataset.loc[fil, 'EDUCATION'] = 4
    dataset['EDUCATION'].value_counts(dropna = False)
    dataset.loc[dataset.MARRIAGE == 0, 'MARRIAGE'] = 3
    
    fil = (dataset.PAY_1 == -1) | (dataset.PAY_1==-2)
    dataset.loc[fil,'PAY_1']=0
    dataset.PAY_1.value_counts()
    fil = (dataset.PAY_2 == -1) | (dataset.PAY_2==-2)
    dataset.loc[fil,'PAY_2']=0
    dataset.PAY_2.value_counts()
    fil = (dataset.PAY_3 == -1) | (dataset.PAY_3==-2)
    dataset.loc[fil,'PAY_3']=0
    dataset.PAY_3.value_counts()
    fil = (dataset.PAY_4 == -1) | (dataset.PAY_4==-2)
    dataset.loc[fil,'PAY_4']=0
    dataset.PAY_4.value_counts()
    fil = (dataset.PAY_5 == -1) | (dataset.PAY_5==-2)
    dataset.loc[fil,'PAY_5']=0
    dataset.PAY_5.value_counts()
    fil = (dataset.PAY_6 == -1) | (dataset.PAY_6==-2)
    dataset.loc[fil,'PAY_6']=0
    dataset.columns = dataset.columns.map(str.lower)
     
    #Standardize the numerical columns
    col_to_norm = ['limit_bal', 'age', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4',
           'bill_amt5', 'bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3',
           'pay_amt4', 'pay_amt5', 'pay_amt6']
    dataset[col_to_norm] = dataset[col_to_norm].apply(lambda x : (x-np.mean(x))/np.std(x))
    
    #Split the data into training and test set
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 1)
    
    return x_train, y_train, x_test, y_test
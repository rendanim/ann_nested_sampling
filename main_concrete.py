import utilities as u
import dynesty
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
import csv  
import pandas as pd
from multiprocessing import Pool, cpu_count
import concurrent.futures
import numpy as np

ITERATIONS = 20000

# the data, split between train and test sets
random.seed(1)
x_train, y_train, x_test, y_test = u.return_concrete_processed_data()


# write out train data
diabetes_train = r'concrete_data_train.csv'
df_train = pd.DataFrame(data=x_train)
df_train[8] = y_train 
df_train.to_csv(diabetes_train, index = False, header = False)

# write out test data
diabetes_test = r'concrete_data_test.csv'
df_test = pd.DataFrame(data=x_test)
df_test[8] = y_test
df_test.to_csv(diabetes_test, index = False, header = False)


logz_file_name = r'logz_concrete.csv'
mse_file_name = r'mse_concrete.csv'

#Store results
x_axis = range(1, 31, 1) # Number of hidden units
sims = range(1, 11, 1) # How many times to run the simulations

#Architecture for mlp
input_neurons = x_train.shape[1] # inputs
output_neurons = 1 # outputs

# Read in alpha and beta from GA results
alpha = pd.read_excel("alpha_concrete_mean.xlsx").values
beta = pd.read_excel("beta_concrete_mean.xlsx").values

alpha = alpha.reshape(30,1)
beta = beta.reshape(30,1)

def prior_ptform(uTheta,*ptform_args):
    alpha_val = ptform_args[0]
    theta = norm.ppf(uTheta, loc = 0, scale = 1/alpha_val)
    return theta

def log_likelihood(W, *logl_args):
    y_pred = u.regression_predictions(x_train, W, 
                           logl_args[0], 
                           logl_args[1], 
                           logl_args[2])
    mse = -u.mean_squared_error(y_pred, y_train)
    beta_val = logl_args[3]
    return mse #* beta_val

def calc_evidence_and_mse(i):
    hidden_neurons = i
    ndim_1 =  (input_neurons + 1) * (hidden_neurons) + (hidden_neurons + 1) * output_neurons
    nlive = 100 * ndim_1
    logl_args = [input_neurons, hidden_neurons, output_neurons, beta[29][0]]
    ptform_args = [alpha[29][0]]
   
    sampler = dynesty.DynamicNestedSampler(log_likelihood,
                               prior_ptform,
                               ndim = ndim_1,
                               nlive = nlive,
                               logl_args = logl_args,
                               ptform_args = ptform_args
                               )
    sampler.run_nested(print_progress = False, 
                       maxbatch = 0,
                       maxiter = ITERATIONS,
                       nlive_init = nlive)
    res = sampler.results
    
    # return the log evidence
    logZ_train = res.logz[-1]
  
    #return test mse

    samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
    index_max = np.argmax(weights)
 
    W_mode = samples[index_max]
    y_pred_mode = u.regression_predictions(x_test, W_mode, input_neurons,hidden_neurons, output_neurons)  
    mse_test = u.metrics.mean_squared_error(y_test, y_pred_mode)
  
    return logZ_train , mse_test


def run_sims(s):
    output1 = list()
    output2 = list()
    
    start = datetime.now()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for logZ_train, mse_test in executor.map(calc_evidence_and_mse, x_axis):
            # put results into correct output list
            output1.append(logZ_train)
            output2.append(mse_test)

    with open(logz_file_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(output1)
    with open(mse_file_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(output2)
    
    end = datetime.now()
    return end - start

def main():
    
    for s  in sims:
        print("\n SIMULATION NUMBER :::::::::::::::::::::", s, "::::::::::::::::::::::::\n")
        output1 = list()
        output2 = list()
        
        start = datetime.now()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for logZ_train, mse_test in executor.map(calc_evidence_and_mse, x_axis):
                # put results into correct output list
                output1.append(logZ_train)
                output2.append(mse_test)
    
        with open(logz_file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(output1)
        with open(mse_file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(output2)
    
        end = datetime.now()
        print("\n Time Taken:::", end - start)
        
    #with concurrent.futures.ThreadPoolExecutor() as executor:
       # for time in executor.map(run_sims, sims):
           # print("Time taken for arbitrary sim",time)
            

if __name__ == '__main__':
    main()
    
    '''Plots '''
    #read in data stored in the files 
    data_logz = pd.read_csv("logz_concrete.csv", header = None) 
    logz_train_avarage = data_logz.mean()
    
    with open(r'logz_concrete_mean.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(logz_train_avarage)
    
    data_mse = pd.read_csv("mse_concrete.csv", header = None) 
    mse_test_avarage = data_mse.mean()
    
    with open(r'mse_concrete_mean.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(mse_test_avarage)
        
    # plot data set
    start = 1
    end = 31
    x_axis = range(start, end, 1)     
        
    plt.style.use(['bmh'])
    fig, ax = plt.subplots(1)
    fig.suptitle('Training Dataset', fontsize = 16)
    ax.set_xlabel('Number of hidden neurons')
    ax.set_ylabel('Average log evidence')
    plt.plot(x_axis, logz_train_avarage[start-1:end-1], '-o')
    plt.show()
    fig.savefig('results_plots/concrete_log_evidence.png')
    
    plt.style.use(['bmh'])
    fig, ax = plt.subplots(1)
    fig.suptitle('Test Dataset', fontsize = 16)
    ax.set_xlabel('Number of hidden neurons')
    ax.set_ylabel('Average Test MSE')
    plt.plot(x_axis, mse_test_avarage[start-1:end-1], '-o')
    plt.show()
    fig.savefig('results_plots/concrete_mse.png')
    
    print(logz_train_avarage)


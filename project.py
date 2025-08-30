# Heart disease probability prediction program based on the input of patient cardiac data

import numpy as np
import pandas as pd

# IMPORTANT VARIABLES
# m - number of iterations in x
# model - updated model (n x 1)
# x - numpy matrix containing the dataset data (m x n) 
# y - binary result associated with each instance of x (m x 1)
# alpha - training rate (starting test with default 0.01)
# grad - gradient of the cross entropy loss function
# heart_disease_predict - logistic regression model for the prediction of heart disease risk

def sigmoid(z, dec=3):
    return 1.0/(1.0 + np.exp(-z))

def h(model,x):
    return sigmoid(x @ model)

# Calculate the gradient of the cross entropy loss function
def get_gradient(model,x,y):
    m = np.shape(x)[0] # m is the number of instances, in this case, the lines of the data frame that originated x
    grad = (np.transpose(x) @ (h(model,x)-y))/m
    return grad

# Calculates the cross entropy loss function of a given model, trained with x vector and y vector for labels
def entropy_loss(model, x, y):
    m = np.shape(x)[0]
    J = []
    for i in range(m):
        j = y[i]*np.log(h(model,x)[i]) + (1-y[i])*np.log(1-h(model,x)[i])
        J.append(j)
    return (-sum(J)/m).item()
        
# Returns the difference between the cross entropy loss function of the previous model and the current 
def dif_cost(prev_model,model,x,y):
    return float(entropy_loss(prev_model,x,y) - entropy_loss(model,x,y))

# Model training - implementing the gradient descent method from scratch to train the model
def log_reg_model(x,y,alpha=0.01,min=0.0001):
    m = np.shape(x)[0]
    col = np.ones((m,1))
    x_int = np.concatenate((col,x), axis=1) # Needed because of the bias (x0)
    model = np.zeros((np.shape((x_int))[1],1)) # Initialize the model as a null matrix with n+1 columns

    # Iterates in a large range and breaks when the difference between the costs of the previous model and the updated one is less than min 
    for i in range(10000):
        prev_model = model
        model = prev_model - alpha*get_gradient(prev_model,x_int,y)
        # print(entropy_loss(prev_model, x_int,y), entropy_loss(model, x_int,y))
        if dif_cost(prev_model, model, x_int, y) < min:
            print(f"Broke log_reg_model loop after {i} iterations") # Just to check if it is breaking before the range
            break
    return model

# Returns data to use for training and separates data from labels, using pandas
def select_train(df,size=10):
    ...

# Implementation of a class that allows manipulation of patient data:
# Handle input
class Patient():
    ...

def main():
    data = pd.read_csv("heart.csv", sep=",")
    patient_data, thal_label = select_train(data)
    heart_disease_risk_predict = log_reg_model(patient_data,thal_label)
    # test this with pytest. It would be cool to test accuracy in the future



if __name__ == "__main__":
    main()

import pandas as pd 
import numpy as np
from project import get_gradient

#FEATURES
############################################################################################################
# Feature 1 - Probability of heart disease (I can still use 3 functions in this and be done with it)
# Use pandas to clean data set - remove duplicate data ? remove some columns ?
# Logistic regression
# Probability of heart disease via input
############################################################################################################
# Feature 2 - Check what parameters are not healthy
# Use formulas for the max heart-rate

# Max heart-rate by age formula:
# simple: 220-age
# more accurate Tanaka formula: MHR = 208 - (0.7*age)

# Use pandas categorical data for range ?
############################################################################################################

#STRATEGY
#1 - Structure main file:
# + model training
# + input from patient
# + data treatment and feature implementation

#2 - Start small: interface in the terminal and model training with a very small dataset in the main file

#3 - Submit50: README and video

#4 - Next level: clone repository, make both private, aply larger dataset and tkinter interface, publish it on github

# Test pandas functionalities on the csv file
# https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?resource=download - check this to learn about manipulating heart data

# Testes 1 - pandas
# print("Test pandas on the heart.csv file")
# heart_data = pd.read_csv("heart.csv", sep=",")

# print("Patients with max heart rate above the recommended")
# critical_hr = heart_data.loc[heart_data["thalach"]>208-(0.7*heart_data["age"])]
# print(critical_hr.loc[:, ["age", "thalach", "thal"]])
# print("\n % of patients with max heart rate above rec")
# print(len(critical_hr.index)/len(heart_data.index) * 100, "%")
# print("\n % of patients with max heart rate above rec and a heart problem")
# critical_hr2 = critical_hr.loc[critical_hr["thal"]!=0]
# print(len(critical_hr2.index)/len(critical_hr.index)*100, "%")

# print(len(heart_data.index))

# print(len(heart_data.index), len(set(heart_data.index)))

# a = [1,2,34,3,3,5,7,8,2,3,9,34]
# print(len(a), len(set(a)))

# Teste 2 - commit via pc
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

def main():
    ...
        
if __name__ == "__main__":
    main()
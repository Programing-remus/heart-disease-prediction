# Cardiac health evaluation system and heart disease risk prediction using Logistic Regression model
#### Video Demo:  <URL HERE>
__Description__: This project was developped as my CS50's Introduction to Programming with Python final project and its main objective is to accurately predict the probability of heart disease of a patient based on the input of a set of heart health related parameters, namely cholesterol levels and blood pressure registrations. 
### Features
+ Predict heart disease risk with logistic regression
+ Assess whether each of the inputs is within a healthy range
+ Provide a clearer view of possible risk factors

### Implementation
For the first part of the project, I decided to implement a Logistics Regression Model from scratch using numpy. In order to get a better understanding of the algorithm, I derived the mathematical formulas manually before translating them to code, creating several reusable functions to train any logistic regression models in the future.
The training of the model requires:

+ a matrix, X, with dimentions mxn (with m instances and n parameters)
+ a vector, y, with a binary label (0 or 1) with m values (one for each instance in X)

The goal of the algorithm is to find a set of paramenters (weights) that better represent the relationship between X and y, producing a model that accurately predicts the heart disease risk for new inputs.

#### Training Funcions:
+ sigmoid(z) - takes a numpy array or a scalar and computes its sigmoid
+ h(model, x) - returns the sigmoid of the matrix product of the model matrix and the input (x), representing the predicted probabilities
+ get_gradient(model,x,y) - commputes the gradient od the cross entropy loss funtion of the updated model
+ entropy_loss(model,x,y) - computes the cross entropy loss function of the updated model
+ cost(prev_model, model) - computes the difference between the cross entropy loss function of the previous model and the updated one.
+ log_reg_model(x,y,alpha=0.01,min=0.0001) - Initializes and trains a logistics regression model by iteratively updating weights using the gradient descent method. The loop is interrupted when 10000 iterations are completed or when the cost function returns a value smaller than min (default value=0.0001)** and the function returns a numpy array with the weights that correspond to the Logistic Regression model.

##### Notes:
+ The learning rate, "alpha", default value was chosen based on research and later validated by accuracy tests runned on the model.
+ The stopping criterion "min" ensures the loop breaks when the value returned by the cost function is no longer significant

#### Dataset
The dataset used was extracted from https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?resource=download and then edited using pandas.

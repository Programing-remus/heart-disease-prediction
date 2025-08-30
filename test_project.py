from project import sigmoid, get_gradient, entropy_loss, dif_cost, log_reg_model, select_train
import numpy as np

def test_sigmoid():
    assert sigmoid(0) == 0.5
    assert round(sigmoid(-4),3) == 0.018
    assert list(round(zi,3) for zi in list(map(sigmoid, [0,1]))) == [0.5, 0.731]
    assert np.all(sigmoid(np.zeros((1,2))) == 0.5)
    X = sigmoid(np.ones((1,3)))
    X2 = np.array([np.round(i,3) for i in X])
    assert np.all(X2 == 0.731)

def test_get_gradient():
    model = np.zeros((2,1)) # 2x1
    x = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5,10]]) # 5x2
    y = np.transpose(np.array([[1,1,0,1,0]])) # 5x1
    assert np.shape(get_gradient(model, x, y)) == (2,1)
    model = np.zeros((3,1)) # 3x1
    x = np.array([[1, 6, 2], [2, 7, 2], [3, 8, 2], [4, 9, 2], [5, 10,2]]) # 5x3
    y = np.transpose(np.array([[1,1,0,1,0]])) # 5x1
    assert np.shape(get_gradient(model,x,y)) == (3,1)
    model = np.zeros((2,1)) # 2x1
    x = np.array([[1, 6], [2, 7]]) # 5x2
    y = np.transpose(np.array([[1,0]]))
    assert np.all(get_gradient(model,x,y)==0.25)
    x = np.array([[0.5, 1.2],[1.0, 1.8],[1.5, 2.5],[3.0, 3.2],[3.5, 4.0],[4.0, 4.5]])
    y = np.transpose(np.array([[0, 0, 0, 1, 1, 1]]))
    assert np.shape(get_gradient(model,x,y)) == (2,1)

def test_entropy_loss():
    model = np.zeros((2,1))
    x = np.array([[1, 6], [2, 7]])
    y = np.transpose(np.array([[1,0]]))
    assert entropy_loss(model,x,y) == float((-2*np.log(0.5)/2).item())

def test_log_reg_model():
    x = np.array([[0.5, 1.2], [1.0, 1.8], [1.5, 2.5], [3.0, 3.2], [3.5, 4.0], [4.0, 4.5]])
    x_i = np.concatenate((np.ones((np.shape(x)[0],1)),x), axis=1)
    y = np.transpose(np.array([[0, 0, 0, 1, 1, 1]]))
    assert (x_i @ log_reg_model(x,y))[0] < 0.001
    assert (x_i @ log_reg_model(x,y))[3] > 0.999
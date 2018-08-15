---
title: Linear Regression - Programming Exercise 1
authors:
- Shashank Jatav
tags:
- coursera
- andrew
- machine_learning
created_at: 2018-08-15 00:00:00
updated_at: 2018-08-15 15:28:14.713716
tldr: This is a python implementation of the week 2 exercise in Andrew ng's course
  for machine learning
thumbnail: images/output_12_0.png
---
Machine Learning Online Class - Exercise 1: Linear Regression

Instructions
------------

This file contains code that helps you get started on the
linear exercise. You will need to complete the following functions
in this exericse:

     warmUpExercise
     plotData
     gradientDescent
     computeCost
     gradientDescentMulti
     computeCostMulti
     featureNormalize
     normalEqn

For this exercise, you will not need to change any code in this file,
or any other files other than those mentioned above.


## warmUpExercise


```python
import numpy as np
```

```python
def warmUpExercise():
    return(np.identity(5))
```

```python
warmUpExercise()
```




    array([[1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.]])



## Plotting the data


```python
## Loading the data 

data_ext1 = np.loadtxt("../ex1/ex1data1.txt", delimiter= ",")
```

```python
data_ext1[0:4]
```




    array([[ 6.1101, 17.592 ],
           [ 5.5277,  9.1302],
           [ 8.5186, 13.662 ],
           [ 7.0032, 11.854 ]])




```python
X = np.c_[np.ones(data_ext1.shape[0]),data_ext1[:,0]]
X[0:4]

# A column of ones has been added to accomodate the theta0 term (intercept term)
```




    array([[1.    , 6.1101],
           [1.    , 5.5277],
           [1.    , 8.5186],
           [1.    , 7.0032]])




```python
y = np.c_[data_ext1[:,1]]
y[0:4]
```




    array([[17.592 ],
           [ 9.1302],
           [13.662 ],
           [11.854 ]])




```python
import matplotlib.pyplot as plt
```

```python
plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s');
```


![png](images/output_12_0.png)



```python
import dill
filename = 'globalsave.pkl'
dill.dump_session(filename)
```
## Gradient Descent

### Compute Cost


```python
def computeCost(X, y, theta):
    """
    COMPUTECOST Compute cost for linear regression
    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    
    # We need to return the following variables correctly 
    
    J=0
    
    """
    ====================== OUR CODE HERE ======================
    Instructions: Compute the cost of a particular choice of theta
    We should set J to the cost.
    """
    
    J = np.sum(np.square(np.matmul(X,theta)-y))/(2*m)
    
    
    return(J)
    
```

```python
computeCost(X,y,theta=[[0],[0]])
```




    32.072733877455676




```python
# Checking compute cost for one more value

computeCost(X,y,theta=[[-1],[2]])
```




    54.24245508201238

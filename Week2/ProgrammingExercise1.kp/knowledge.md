---
title: Linear Regression - Programming Exercise 1
authors:
- Shashank Jatav
tags:
- coursera
- andrew
- machine_learning
created_at: 2018-08-15 00:00:00
updated_at: 2018-08-15 19:12:15.747738
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
filename = 'ProgrammingExercise1.pkl'
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

## Expected cost value (approx) 32.07
```




    32.072733877455676




```python
# Checking compute cost for one more value

computeCost(X,y,theta=[[-1],[2]])

## Expected cost value (approx) 54.24, pretty close huh !!
```




    54.24245508201238



### Gradient Descent Code


```python
def gradientDescent(X, y, theta, alpha, num_iters):
    """
    GRADIENTDESCENT Performs gradient descent to learn theta
    theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    """
    
    m = y.shape[0]  ## Number of training examples
    J_history = np.zeros(num_iters)  ## Initializing J to 0
    
    ## Starting the iterations
    
    for iteration in np.arange(num_iters):
        """
        ====================== OUR CODE HERE ======================
        Instructions: Perform a single gradient step on the parameter vector theta
        """
        
        theta = theta - (alpha/m)*np.transpose(np.matmul(np.transpose(np.matmul(X,theta)-y), X))
        
        ## Please note the importance of adding intercept term to X in the beginning
        
        J_history[iteration] = computeCost(X, y, theta)
        
    return(theta, J_history)
    
```

```python
## Check theta values as in the assignment

theta , Cost_Iterations = gradientDescent(X, y, theta=[[0],[0]], alpha=0.01, num_iters=1500)

print theta

## Expected theta values (approx): -3.6303  1.1664
```
    [[-3.63029144]
     [ 1.16636235]]



```python
plt.plot(Cost_Iterations)
plt.ylabel('Cost at each iteration')
plt.xlabel('Iterations');
```


![png](images/output_22_0.png)



```python
# Predict values for population sizes of 35,000 and 70,000

predict1 = np.matmul([1, 3.5] ,theta)

predict1
```




    array([0.45197679])




```python
predict2 = np.matmul([1, 7] ,theta)

predict2
```




    array([4.53424501])



## Visualizing the results


```python
## Plotting the fitted line
# Getting the relevant x and y coordinates

xcoord = np.linspace(5,23)
ycoord = theta[0] + theta[1]*xcoord # this is simply y = mx + c

# Plot gradient descent
plt.scatter(X[:,1], y, s=30, c='r', linewidths=1)
plt.plot(xcoord,ycoord, label='Linear regression Gradient descent')
```




    [<matplotlib.lines.Line2D at 0x10da00f90>]





![png](images/output_26_1.png)


### Visualizing J(theta_0, theta_1)


```python
from mpl_toolkits.mplot3d import axes3d
```

```python
# Grid over which we will calculate J

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

Zvalues_J = np.zeros((theta0_vals.shape[0],theta1_vals.shape[0]))
xxcoord, yycoord = np.meshgrid(theta0_vals, theta1_vals, indexing='xy') # my favourite function which has no such counterpart in R or maybe

# Calculate Z-values (Cost) based on grid of coefficients
for (i,j),v in np.ndenumerate(Zvalues_J):
    Zvalues_J[i,j] = computeCost(X,y, theta=[[xxcoord[i,j]], [yycoord[i,j]]])
    
fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Left plot
CS = ax1.contour(xxcoord, yycoord, Zvalues_J, np.logspace(-2, 3, 20), cmap=plt.cm.jet)
ax1.scatter(theta[0],theta[1], c='r')

# Right plot
ax2.plot_surface(xxcoord, yycoord, Zvalues_J, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
ax2.set_zlabel('Cost')
ax2.set_zlim(Zvalues_J.min(),Zvalues_J.max())
ax2.view_init(elev=15, azim=230)

# settings common to both plots
for ax in fig.axes:
    ax.set_xlabel(r'$\theta_0$', fontsize=17)
    ax.set_ylabel(r'$\theta_1$', fontsize=17)
```


![png](images/output_29_0.png)



```python
# Saving checkpoint
import dill
filename = 'ProgrammingExercise1.pkl'
dill.dump_session(filename)
```

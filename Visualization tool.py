# This tool aims to provide a simple visualization for low-dimensional optimization problems.

# The search space represents the domain in which samples may be drawn from
# The objective function is the target which we are trying to minimize or maximise.
# It will return a single value relating to the performance of its provided inputs
# The n-dimensional space must be sampled adequately

# The surrogate function will be created with the goal of improving computational efficiency

# The acquisition function is the tool that leverages the posterior to select the next sample query point

# The algorithm may be summarized through the following steps
#   1. Collect initial sample data
#   2. Use the objective function to evaluate the samples
#   3. Update the observation dataset, which in turn changes the surrogate model
#   4. Collect a sample
#   5. Repeat 2 -> 4
#   6. Provide optimized solution

####################################
#   EXAMPLE: 1. ONE-DIMENSIONAL    #
####################################
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor


# Define the objective function. Note, this will typically be a black-box function.
# If we have access to gradient information, gradient-based solvers will be a better approach.
# For this example, a modified sin function will be used. This function has 5 peaks.
def objective_function(x):
    y = x**2 * math.sin(5*math.pi*x)**6
    return y

# Let's visualise this curve between x=0 and x=1
X = np.linspace(0,1,1000)
y = [objective_function(x) for x in X]
plt.plot(X,y)

# To test our algorithm, lets first find the true maximum solutions
# Note that in practise, we would not have access to this information
x_best_index = np.argmax(y)

print('Location of global max: x = %.2f, y = %.2f' % (X[x_best_index], y[x_best_index]))
print(y[np.argmax(y)])

plt.plot(X[x_best_index],y[np.argmax(y)], marker="o", markersize=5, color = "red")
plt.show()

# We now have a test set-up. Let's create our proxy function
# The Gaussian process is expressed as:  f(x) -- GP(m(x),K(x,x′))

# First, we need to select our Kernel function K (also known as the covariance function)
# The Kernel function calculates the GP’s covariance between data points
# By default, the prior mean is assumed to be constant and zero
model = GaussianProcessRegressor()

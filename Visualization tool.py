# Importing Packages
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics.pairwise import euclidean_distances
import scipy
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import norm
from warnings import catch_warnings
from warnings import simplefilter
import random

# Set matplotlib and seaborn plotting style
sns.set_style('darkgrid')
np.random.seed(20)

# This tool aims to provide a simple visualization for low-dimensional optimization problems.

# The search space represents the domain in which samples may be drawn from
# The objective function is the target which we are trying to minimize or maximise.
# It will return a single value relating to the performance of its provided inputs
# The n-dimensional space must be sampled adequately

# The surrogate function will be created with the goal of improving computational efficiency

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

# Define the objective function. Note, this will typically be a black-box function.
# If we have access to gradient information, gradient-based solvers will be a better approach.
# For this example, a modified sin function will be used.
def objective_function(x):
    #y = x**2 * math.sin(5*math.pi*x)**6
    y = x/5 + (x*math.sin(x))
    return y

# Let's visualise this curve between x=-6 and x=6
X = np.linspace(-6,6,1000)
y = [objective_function(x) for x in X]
plt.plot(X,y,label="x/5 + xsin(x)")

# To test our algorithm, lets first find the true maximum solution
# Note that in practise, we would not have access to this information
x_best_index = np.argmax(y)

print('Location of global max: x = %.2f, y = %.2f' % (X[x_best_index], y[x_best_index]))
print(y[np.argmax(y)])

plt.plot(X[x_best_index],y[np.argmax(y)], marker="o", markersize=5, color = "red", label="Maxima")
plt.title((
    'Target Function f(x) = x/5 + xsin(x)'))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.legend(loc="upper left")
plt.show()

# We now have a test set-up. Let's create our proxy function
# The Gaussian process is expressed as:  f(x) -- GP(m(x),K(x,x′))

# First, we need to select our Kernel function K (also known as the covariance function)
# The Kernel function calculates the GP’s covariance between data points


######################################
#    DEFINITION: KERNEL FUNCTION     #
######################################

# Define the kernel function: exponentiated quadratic
# This function returns the square-root of the distance between xa and xb * (-0.5)
def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with σ=1"""

    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)


######################################
#   VISUALIZATION: KERNEL FUNCTION   #
######################################

# Let's visualise the kernel function.
# Diagonal entries = the variance
# Non-diagonal entries = the covariance between variables
def visualise_kernel():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    xlim = (-3, 3)
    # Expand the shape of the array
    X = np.expand_dims(np.linspace(*xlim, 25), 1)
    # Calculate the covariance matrix
    Σ = exponentiated_quadratic(X, X)
    # Plot covariance matrix
    im = ax1.imshow(Σ, cmap=cm.YlGnBu)
    cbar = plt.colorbar(
        im, ax=ax1, fraction=0.045, pad=0.05)
    cbar.ax.set_ylabel('$k(x,x)$', fontsize=10)
    ax1.set_title((
        'Exponentiated quadratic \n'
        'covariance matrix'))
    ax1.set_xlabel('x', fontsize=13)
    ax1.set_ylabel('x', fontsize=13)
    ticks = list(range(xlim[0], xlim[1]+1))
    ax1.set_xticks(np.linspace(0, len(X)-1, len(ticks)))
    ax1.set_yticks(np.linspace(0, len(X)-1, len(ticks)))
    ax1.set_xticklabels(ticks)
    ax1.set_yticklabels(ticks)
    ax1.grid(False)

    # Show covariance with X=0
    xlim = (-4, 4)
    X = np.expand_dims(np.linspace(*xlim, num=50), 1)
    zero = np.array([[0]])
    Σ0 = exponentiated_quadratic(X, zero)
    # Make the plots
    ax2.plot(X[:,0], Σ0[:,0], label='$k(x,0)$')
    ax2.set_xlabel('x', fontsize=13)
    ax2.set_ylabel('covariance', fontsize=13)
    ax2.set_title((
        'Exponentiated quadratic  covariance\n'
        'between $x$ and $0$'))
    # ax2.set_ylim([0, 1.1])
    ax2.set_xlim(*xlim)
    ax2.legend(loc=1)

    fig.tight_layout()
    plt.show()

visualise_kernel()




######################################
#   EXAMPLE: SAMPLING FROM PRIOR     #
######################################

def prior_samples():
    # Let's choose how many points we want to sample for each function
    prior_sampling = 50
    # Let's choose how many functions we want to draw
    no_functions = 6
    # Let's create a regular interval of points to draw
    X_example = np.linspace(-6,6,prior_sampling)
    # Reshape the array into a 50x1
    X_example = np.expand_dims(X_example,1)
    # Calculate the kernel function output for each pair within X and X
    Σ = exponentiated_quadratic(X_example,  X_example)  # Kernel of data points

    # Randomly sample the prior at the points
    # Mean function is an array of zeros for each data point
    # Kernel function has been calculated
    # .multivariate_normal will draw random samples
    y_example = np.random.multivariate_normal(
    mean=np.zeros(prior_sampling), cov=Σ,
    size= no_functions)

    # Each array in Y is one of the functions to be plotted
    for i in range(no_functions):
        plt.plot(X_example, y_example[i], linestyle='-', marker='o', markersize=3)

    plt.title((
        '6 different function realizations at 50 points\n'
        'sampled from prior: GP with exponentiated quadratic kernel'))
    plt.xlim([-6, 6])
    plt.show()

prior_samples()


######################################
#   APPLICATION: GAUSSIAN PROCESS    #
######################################

# Let's first define our initial sampling first
n1 = 8  # Number of points to condition on (training points)
#n2 = 75  # Number of points in posterior (number of test points)
n2 = 200
domain = (-6, 6)

# Sample observations (X1, y1) on the function
X1 = np.random.uniform(domain[0] + 2, domain[1] - 2, size=(n1, 1))
y1 = np.asarray([objective_function(x) for x in X1])
y1 = np.reshape(y1, n1)

# Let's now define the uniform prediction points to capture function
X2 = np.linspace(domain[0], domain[1], n2).reshape(-1, 1)

# Gaussian process posterior
def GP(X1, y1, X2, kernel_func):
    """
    Calculate the posterior mean μ2 and posterior covariance matrix Σ2 for y2
    based on the corresponding input X2, the observations (X1, y1),
    and the prior kernel function.
    """

    print("____SHAPE X1____")
    print(X1.shape)
    print(X1)
    print("____SHAPE X2____")
    print(X2.shape)
    print(X2)

    # Kernel of the observations
    Σ11 = kernel_func(X1, X1)
    # Kernel of observations vs to-predict
    Σ12 = kernel_func(X1, X2)
    # Σ12 Σ22 ^-1
    solved = scipy.linalg.solve(Σ11, Σ12, assume_a='pos').T

    # Compute posterior mean
    # μ2 = 0 + Σ12 Σ22 ^-1 (y1 - 0)
    # @ is simply used for matrix multiplication
    μ2 = solved @ y1
    # Compute the posterior covariance
    # Σ2 = Σ22 - Σ12 Σ22 ^-1 Σ12
    Σ22 = kernel_func(X2, X2)
    Σ2 = Σ22 - (solved @ Σ12)
    return μ2, Σ2  # mean, covariance

# Compute the posterior mean and covariance
def posterior_mean_and_covariance(X1, y1, X2):

    # Compute posterior mean and covariance
    μ2, Σ2 = GP(X1, y1, X2, exponentiated_quadratic)
    # Compute the standard deviation at the test points to be plotted
    # Standard deviation is the square root of diagonal entries of Σ
    σ2 = np.sqrt(np.diag(Σ2))

    # Draw some samples of the posterior
    ny = 5  # Number of functions that will be sampled from the posterior
    y2 = np.random.multivariate_normal(mean=μ2, cov=Σ2, size=ny)

    # Plot the posterior distribution and some samples
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(6, 6))

    graphs = np.asarray([objective_function(x) for x in X2])
    graphs = np.reshape(graphs, len(X2))

    # Plot the distribution of the function (mean, covariance)
    ax1.plot(X2, graphs, 'b--', label='$x/5 + xsin(x)$')
    ax1.fill_between(X2.flat, μ2-2*σ2, μ2+2*σ2, color='red',
                     alpha=0.15, label='$2 \sigma_{2|1}$')
    ax1.plot(X2, μ2, 'r-', lw=2, label='$\mu_{2|1}$')
    ax1.plot(X1, y1, 'ko', linewidth=2, label='$(x_1, y_1)$')
    ax1.set_xlabel('$x$', fontsize=13)
    ax1.set_ylabel('$y$', fontsize=13)
    ax1.set_title('Distribution of posterior and prior data.')
    ax1.axis([domain[0], domain[1], -10, 10])
    ax1.legend()
    # Plot some samples from this function
    ax2.plot(X2, y2.T, '-')
    ax2.set_xlabel('$x$', fontsize=13)
    ax2.set_ylabel('$y$', fontsize=13)
    ax2.set_title('5 different function realizations from posterior')
    ax1.axis([domain[0], domain[1], -10, 10])
    ax2.set_xlim([-6, 6])
    plt.tight_layout()
    plt.show()

    # Let's record our best result
    best_μ = np.amax(μ2)

    return best_μ,X2, μ2
prev_best,X_new,μ_data = posterior_mean_and_covariance(X1, y1, X2)


######################################
#   INTRODUCE ACQUISITION FUNCTION   #
######################################

def acquisition_func(X1,y1, X2, prev_best):

    # Compute posterior mean and covariance
    μ2, Σ2 = GP(X1, y1, X2, exponentiated_quadratic)
    std = np.sqrt(np.diag(Σ2))

    # Plug in the PI formula to get probability of improvement at the samples.
    # We add a (very) small number to std to avoid dividing by zero
    P_I = (μ2 - prev_best)/(std + 1E-10)
    # Let's calculate the cumulative probability of every entry in P_I
    P_I = norm.cdf(P_I)
    return P_I


###########################################
#   PLOTTING PROBABILITY OF IMPROVEMENT   #
###########################################

n1 = 8  # Number of points to condition on (training points)
n2 = 75  # Number of points in posterior (test points)
domain = (-6, 6)


prob_improve = acquisition_func(X1,y1, X2, prev_best)
# Get max index
index = np.argmax(prob_improve)
print("Index is:")
print(prob_improve)

# Plot the distribution of the function (mean, covariance)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 7))
graphs = np.asarray([objective_function(x) for x in X2])
graphs = np.reshape(graphs, len(X2))

ax1.plot(X2, graphs, 'b--', label='$x/5 + xsin(x)$')
ax1.plot(X1, y1, 'ko', linewidth=2, label='$(x_1, y_1)$')
ax1.plot(X2[index], objective_function(X2[index]), marker="o", markersize=5, color = "red")
ax1.set_xlabel('$x$', fontsize=13)
ax1.set_ylabel('$y$', fontsize=13)
ax1.set_title('Proxy model')
ax1.axis([domain[0], domain[1], -10, 10])
ax1.legend()

ax2.plot(X2, prob_improve, 'b-', label='$PI$')
ax2.plot(X2[index], prob_improve[index],'ko', linewidth=2,)
ax2.set_xlabel('$x$', fontsize=13)
ax2.set_ylabel('$PI$', fontsize=13)
ax2.set_title('Probability of Improvement')
ax2.axis([domain[0], domain[1], -0.8, 0.8])
ax2.legend()

plt.show()


#############################################
#   ITS TIME TO LOOP AND UPDATE OUR PROXY   #
#############################################

# For ease of use, lets create a larger dataset
X_data = X1
Y_data = y1



# Let's put this all together and iterate 100 times
for i in range(5):

    print("Iteration: ")
    print(i)

    prob_improve = acquisition_func(X1, y1, X2, prev_best)
    index = np.argmax(prob_improve)


    # We now know what point we want to query. Let's check its true value
    print("Index is:")
    print(index)
    print(prob_improve)

    y_new = objective_function(X2[index])

    # First, lets see if we're finished our optimization
    if y_new in y1[:]:
        break

    # Let's add this new information to our dataset
    X1 = np.append(X1,X2[index])
    X1 = np.reshape(X1,(X1.size,1))
    y1 = np.append(y1,y_new)

    print(y1)
    print(X1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 7))
    ax1.plot(X,y)
    ax1.plot(X1, y1, 'ko', linewidth=2, label='$(x_1, y_1)$')
    plt.show()

print("Maximum: x = %0.3f, y = %0.3f" % (np.amax(X1), np.amax(y1)))
plt.plot(X,y)
plt.plot(X1[np.argmax(y1)],np.amax(y1), marker="o", markersize=5, color = "red")
plt.show()
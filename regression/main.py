import matplotlib.pyplot as plt
import numpy as np

from regression.rbf import reward
from regression.es import SimpleGA

MAX_ITERATION = 20
N_POPULATION = 10

fit_func = reward


def test_solver(solver):
    history = None
    best = None
    for j in range(MAX_ITERATION):

        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)
        for i in range(solver.popsize):
            cent = np.reshape(solutions[i][:-1], (K_cent, X_train.shape[1]))
            sigma = solutions[i][-1]
            error, y_pred = fit_func(cent, sigma, X_train, y_train)
            fitness_list[i] = error
            solver.tell(fitness_list)
            result = solver.result()  # first element is the best solution, second element is the best fitness
            history = y_pred
            best = result[1]
        print("best fitness at iteration", (j + 1), best)

    return history


np.random.seed(10)
NUM_SAMPLES = 100
X_train = np.random.uniform(0., 1., NUM_SAMPLES)
X_train = np.sort(X_train, axis=0)
noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
y_train = np.sin(2 * np.pi * X_train) + noise

X_train = np.reshape(X_train, (NUM_SAMPLES, 1))
y_train = np.reshape(y_train, (NUM_SAMPLES, 1))

K_cent = 5
N_PARAMS = K_cent * X_train.shape[1] + 1

ga = SimpleGA(N_PARAMS,                # number of model parameters
              popsize=N_POPULATION,   # population size
              sigma_init=0.5,
              )

ga_history = test_solver(ga)
plt.plot(X_train, y_train, '-o', label='true')
plt.plot(X_train, ga_history, '-o', label='RBF-Net')
plt.legend()
plt.tight_layout()
plt.show()

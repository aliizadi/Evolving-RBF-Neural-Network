import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from multilableClassifiction.rbf import reward
from multilableClassifiction.rbf import predict
from multilableClassifiction.es import SimpleGA

MAX_ITERATION = 20
N_POPULATION = 10

fit_func = reward


def test_solver(solver):
    train_pred = None
    best_fitness = None
    best_cent = None
    best_sigma = None
    best_W = None
    for j in range(MAX_ITERATION):

        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)
        for i in range(solver.popsize):
            cent = np.reshape(solutions[i][:-1], (K_cent, X_train.shape[1]))
            sigma = solutions[i][-1]
            error, y_pred, best_W= fit_func(cent, sigma, X_train, y_train)
            fitness_list[i] = error
            solver.tell(fitness_list)
            result = solver.result()  # first element is the best solution, second element is the best fitness
            train_pred = y_pred
            best_fitness = result[1]
            best_cent = cent
            best_sigma = sigma

        print("best fitness at iteration", (j + 1), best_fitness)

    return train_pred, best_cent, best_sigma, best_W


colors = ['b', 'r', 'g']

k_class = 3

centers_ = [[4, 2],
            [1, 7],
            [5, 6]]

sigmas = [[0.8, 0.3],
          [0.3, 0.5],
          [1.1, 0.7]]

np.random.seed(42)
x = np.zeros(1)
y = np.zeros(1)
labels = np.zeros(1)
for i, ((xmu, ymu), (x_sigma, y_sigma)) in enumerate(zip(centers_, sigmas)):
    x = np.hstack((x, np.random.standard_normal(100) * x_sigma + xmu))
    y = np.hstack((y, np.random.standard_normal(100) * y_sigma + ymu))
    labels = np.hstack((labels, np.ones(100) * i))


X = np.column_stack((x, y))
y = labels

one_hot_labels = np.zeros((y.shape[0], k_class))

for i in range(y.shape[0]):
    one_hot_labels[i, int(labels[i])] = 1
y = one_hot_labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

fig0, ax0 = plt.subplots()
for label in range(3):
    ax0.plot(X_train[:, 0][np.array([np.where(r == 1)[0][0] for r in y_train]) == float(label)],
             X_train[:, 1][np.array([np.where(r == 1)[0][0] for r in y_train]) == float(label)],
             '.', color=colors[label])


K_cent = 4
N_PARAMS = K_cent * X_train.shape[1] + 1

ga = SimpleGA(N_PARAMS,                # number of model parameters
              popsize=N_POPULATION,   # population size
              sigma_init=0.5,
              sigma_decay=0.999,  # anneal standard deviation
              sigma_limit=0.01,  # stop annealing if less than this
              forget_best=True
              )


ga_train_pred, ga_cent, ga_sigma, ga_W = test_solver(ga)

test_pred = predict(cent=ga_cent, sigma=ga_sigma, W=ga_W, X_test=X_test)


fig1, ax1 = plt.subplots()

for label in range(3):
    ax1.plot(X_test[:, 0][np.array(test_pred) == float(label)], X_test[:, 1][np.array(test_pred) == float(label)],
             '.', color=colors[label])

print('accuracy: ', accuracy_score([np.where(r == 1)[0][0] for r in y_test], test_pred))

plt.show()
plt.cla()

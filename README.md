# Evolving RBF Neural Network

Implementation of RBF neural network trained using Evolutionary strategy as genetic algorithms for two regression and multi label classification. This is final project of computaional intelligence course under supervision of [Dr.Mohammad Mehdi Ebadzadeh](https://ceit.aut.ac.ir/autcms/people/verticalPagesAjax/professorHomePage.htm?url=ebadzadeh&depurl=computer-engineering&lang=en&cid=3967751)

## RBF Neural Network

#### Architecture
RBF neural netowork contains three layer using RBF function as activation function for hidden layer.
A 3 layer neural network with linear activation function can just classify a linearly separable data points but RBF neural network can classify non-linear seoarable data.
![rbf architecture](https://github.com/aliizadi/Evolving-RBF-Neural-Network/blob/master/media/1.png)


### Activation Function
#### RBF Function:
![rbf function](https://github.com/aliizadi/Evolving-RBF-Neural-Network/blob/master/media/2.png)
This function has two parameters vi as ceneter and Sigma. Given these two parameters this function acitvates when input has an euclidean distance near to center. So every neuron in layer two is representive of points near it. And all these neurons decide together in next layer how much an input can affect in total error. 

## Problem
Problem here is to find weights between layer one and two which are centers of rbf functions and weights of between layer two and three which used for tasks of classifcation and regression.


## Solution
Instead of gradient descent we use genetic algorithm to find centers and weights.
1. find rbf centers by evoltionary strategies
2. compute network error by giving input data to network and found weights in step 1.
3. repeat step 1 and 2 until computed error by provided centers by Es becomes minimum.

### Evolutionary Strategies
Black-box optimization: Intuitively, the optimization is a “guess and check” process, where we start with some random parameters and then repeatedly 
1) tweak the guess a bit randomly, and 
2) move our guess slightly towards whatever tweaks worked better.
* More description in [openai blog](https://openai.com/blog/evolution-strategies/)
* I used ES implementation from [estool](https://github.com/staturecrane/PyTorch-ES)
Gernal Solution used for ES is based on below algorithms. Thanks to [hardmaru](https://github.com/hardmaru)
```
solver = EvolutionStrategy()
while True:

  # ask the ES to give us a set of candidate solutions
  solutions = solver.ask()

  # create an array to hold the solutions.
  # solver.popsize = population size
  rewards = np.zeros(solver.popsize)

  # calculate the reward for each given solution
  # using your own evaluate() method
  for i in range(solver.popsize):
    rewards[i] = evaluate(solutions[i])

  # give rewards back to ES
  solver.tell(rewards)

  # get best parameter, reward from ES
  reward_vector = solver.result()

  if reward_vector[1] > MY_REQUIRED_REWARD:
    break
```


## Regression Result
Here we tests networks to model sin, cos and tan function from sampled data with noise.
First five figures show that how number of neorons in hidden layer as centers can affect to find best model. here we could find the best model with 5 neurons in hidden layer.
* Blue points are sampled data for training and orange ones are predicted points from trained points.
![regression result](https://github.com/aliizadi/Evolving-RBF-Neural-Network/blob/master/media/3.png)
![regression result](https://github.com/aliizadi/Evolving-RBF-Neural-Network/blob/master/media/4.png)

Also you can see models for cos and tan function. Notice that for tan function we used just 2 neurons for hidden layer.
![regression result](https://github.com/aliizadi/Evolving-RBF-Neural-Network/blob/master/media/5.png)


## Classification Result
Because last layer contains just one neurons and we need multilabel classification, first we changed last layer to n neurons as n is number of classes. Then we used softmax function as activation in last layer which shows probability of each neorons belongs to each class. Finally each nearons with maximum probablity betweens 0 and 1 is labeld as predicted class.
Below figures shows classification result for random generated data belongs to 3 classes.
As you can see with 80 percent training data we can correctly predict 20 percent test data with accuracy of 100 percent.  **Notice that this is a 3 layer neural network with rbf activation function and data is non-linear separable**
![classification result](https://github.com/aliizadi/Evolving-RBF-Neural-Network/blob/master/media/6.png)

*  Here is the results of training set for 2 and 3 neurons in hidden layer which result 67 and 99 percent accuracy for training data which shows with more center we can model data betters.

 ![classification result](https://github.com/aliizadi/Evolving-RBF-Neural-Network/blob/master/media/7.png)



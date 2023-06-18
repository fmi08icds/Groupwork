# Reasons and strategies for data creation -> Gerne dazufügen :) 

### Introduction
The goal of this project is to explain (linear) regression in an intuitive way, to somebody who has never seen it before. The standard "real-world" approach, involving the use of a public dataset (e.g. Kaggle) is one that effectively demonstrates the usefulness of performing regression analysis on data, since it provides a regression fit line (from now on fit) which can be visually assessed for its correspondence to the data, i.e. how good the fitting line corresponds to the points. The disadvantage of this method is that the dataset is fixed, meaning that the user cannot see how a regression algorithm responds and performs to changes in the dataset. 

Focussing on the educational approach, with the goal of intuitive learnability, it is important for the user to have the ability to manipulate the input data in various ways and run the regression algorithm on this manipulated data, thereby being able to clearly observe the change in performance and response of the algorithm. 

In this readme, we list and discuss the different data creation methods and parameters that can be tweaked by the user to create a dataset for the regression analysis.  

For simplicity, we describe the 2-dimensional case ( $\bar{y} = b_0 + b_1*x$), where b_0 is the intercept and b_1 is the slope.   


### 1. Specify line (slope and intercept) + specify a distribution + scale

- user provides the slope and intercept of their choice. We denote the equation of this line by $f(x)$. (ideally the regression algorithm will return a fit that is similar to the line provided by the user). 

- user provides a probability distribution and its required parameters for the independent variable $x$. This distribution will determine the arrangement of $x$-values in the dataset. 

- user provides another probability distribution and its required parameters, for the dependent variable $y$. 

***Functionality:*** the generated $x$-values are plugged into the line $f(x)$, provided by the user, yielding points that are directly on the line. These points are then shifted vertically by a distance (positive or negative) provided by a randomly selected value from the probability distribution given for the dependent variable. 

The result is a scatter plot which can be fully determined and scaled by the user. Tweaking the distibution parameters allows the generation of various datasets, which perform differently to regression. This way, the user can intuitively understand the effects of different datasets and how the regression deals with them. 



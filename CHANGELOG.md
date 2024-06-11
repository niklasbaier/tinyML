## v0.5.0 (2024-06-11)

### Feat

- add DecisionTreeClassifier & integration test
- add utility functions to calculate information gain & entropy

### Refactor

- move DecisionTreeClassifier out into own module
- move sample data fixtures into conftest

## v0.4.2 (2024-06-10)

### Refactor

- move BaseLinearRegression in its own file

## v0.4.1 (2024-06-02)

### Refactor

- move logistic regression into classifiers

## v0.4.0 (2024-06-02)

### Feat

- add binary cross-entropy loss function
- add logistic regression & test
- add new metrics & activation function

### Fix

- fix wrong weights & bias gradients and add visualization method for training
- fix ambiguity when checking for uninitialized weights / bias

### Refactor

- slightly move loss (& accuracy) calculation for better readability

## v0.3.2 (2024-06-01)

### Refactor

- remove unneeded folders & .gitkeep files

## v0.3.1 (2024-06-01)

### Refactor

- more concise formulation

## v0.3.0 (2024-06-01)

### Feat

- add Linear Regression with OLS algorithm & corresponding test

### Refactor

- add Base Class for Linear Regression and refactor Gradient Descent & OLS fitting algorithm implementations into proper child classes
- renaming function parameter
- slightly adjust MSE calculation

## v0.2.0 (2024-06-01)

### Feat

- add Linear Regression algorithm (using gradient descent) and corresponding test
- add basic MSE metric with unit tests

### Refactor

- remove unnecessary file
- remove unnecessary files

## v0.1.1 (2024-05-20)

### Refactor

- initial project setup

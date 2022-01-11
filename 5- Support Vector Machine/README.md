# Support Vector Machine
### Problem statement:

The attached dataset "heart.csv" contain 303 records of patients have heart disease or
not according to features in it. You are required to build SVM model to predict whether
patient have heart disease or not (target).


### Update Weights Equations:


1. If Point is correctly classified: w = w - learning rate . (2 * lambda * w)
2. If Point is not correctly classified: w = w + learning rate . (yi . xi - 2 * lambda * w)
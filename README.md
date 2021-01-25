# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset used in this experiment contains data regarding Bank Marketing data. We seek to predict the binary target value whether a client subscribed for a deposit or not. Two approaches are compared to find the best model: 1) tuning the hyperparameters with HyperDrive and 2) selecting the optimal model using Automated Machine Learning (AutoML). The model with the highest accuracy is the best model. 


## Scikit-learn Pipeline
The Bank Marketing data is uploaded via the provided URL. The data is split in train and test data (80/20). Logistic Regression is used to predict the target variable. The following hyperparameters are optimized:

- C: Inverse of regularization strength 
- Max_Iter: Maximum number of iterations to converge

The RandomParameterSampler is used to optimize the hyperparameters, as earlier experiments have shown that this method is relatively time efficient with high performance. 

An early stopping policy is applied to the pipeline in order to terminate poorly performing runs. The policy early terminates any runs where the primary metric is not within the specified slack factor/slack amount with respect to the best performing training run.

## AutoML
Automated Machine Learning (AutoML) is used to automate the entire Machine Learning pipeline. The class AutoMLConfig allows for specifying parameters. The following are used:
- experiment_timeout_minutes: 30 min
- task: classification
- primary_metric: accuracy
- training_data: df
- label_column_name: y-whether cliemt makes term deposit or not
- n_cross_validations: 5

## Pipeline comparison
The highest accuracy using HyperDrive is 0.9165 with C is 4 and max iterations 200. The highest accuracy of AutoML is 0.9167 with an VotingEnsemble algorithm. 

## Future work
Only a limited number of values is included in the RandomParameterSampler, while optimal values may be left out. Furthermore, the chosen termination policy could be altered to experiment whether a more optimal model may result. The data cleaning could be executed differently and feature selection/engineering could be performed different. The number of cross_validation runs could be experimented with. Last but not least, different Machine Learning models could be experimented with. 



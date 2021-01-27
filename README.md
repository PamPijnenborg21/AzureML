# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset used in this experiment contains data regarding Bank Marketing data. We seek to predict the binary target value whether a client subscribed for a deposit or not. Two approaches are compared to find the best model: 1) tuning the hyperparameters with HyperDrive and 2) selecting the optimal model using Automated Machine Learning (AutoML). The model with the highest accuracy is the best model. 


## Scikit-learn Pipeline
The pipeline architecture is constructed as follows: First the parameter search space is defined and the primary metric (i.e. accuracy) to optimize is specified. An early termination policy is applied to early terminate low-performing runs to increase efficiency. Resources are allocated to perform the runs. An experiment is created to run the earlier defined configuration. The results of the training runs are visualized to analyze and from which the best configuration is selected. 

The Bank Marketing data is uploaded via the provided URL. The data is split in train and test data (80/20). Logistic Regression is used to predict the target variable. The following hyperparameters are optimized:

- C: Inverse of regularization strength 
- Max_Iter: Maximum number of iterations to converge

An early stopping policy is applied to the pipeline in order to terminate poorly performing runs. The policy early terminates any runs where the primary metric is not within the specified slack factor/slack amount with respect to the best performing training run. The slack factor is the ratio utilized to calculate the allowed distance from the best performing experiment run. The slack amount is the absolute distance allowed from the best performing run. Early stopping will make the hyperparameter tuning more efficient, as non-promising runs are terminated earlier. 

The RandomParameterSampler is used to optimize the hyperparameters, as earlier experiments have shown that this method is relatively time efficient with high performance. An alternative for the RandomParameterSampler is the GridParameterSampling. The difference between RandomParameterSampler and GridParameterSampling is that the random sampler supports discrate as well as continuous hyperparameters and are randomly defined in the search space. Grid sampling, on the other hand, only support discrete hyperparameters. THe hyperparameters generated with the Grid sampling are defined by performing a grid search over all possible values of hyperparameters. Grid sampling usually is less computational efficient, while having the same performance as the RandomParameterSampler.

## AutoML
Automated Machine Learning (AutoML) is used to automate the entire Machine Learning pipeline. The class AutoMLConfig allows for specifying parameters. The following parameters are used and specified:
- experiment_timeout_minutes: 30 min
- task: classification
- primary_metric: accuracy
- training_data: df
- label_column_name: y-whether client makes term deposit or not
- n_cross_validations: 5

The best performing algorithm is EnsembleVoting, which uses multiple algorithms and the majority vote (in case of a classification problem) is the final outcome. As the EnsembleVoting uses multiple algorithms, it breduces bias regarding a single algorithm and using multiple algorithms makes the algorithm robust.

## Pipeline comparison
The highest accuracy using HyperDrive is 0.9165 with C is 4 and max iterations 200. The highest accuracy of AutoML is 0.9167 with an VotingEnsemble algorithm. The AutoML requires less steps to run experiments and the accuracy of the best model generated by AutoML is also higher compared to the highest accuracy of the best model generated with HyperDrive. 

## Future work
Only a limited number of values is included in the RandomParameterSampler, while optimal values may be left out. So including a wider range of values in the RandomParameterSampler may increase performance. Furthermore, the chosen termination policy could be altered to experiment whether a more optimal model may result, because the used termination policy may not be the optimal one. The data cleaning could be executed differently and feature selection/engineering could be performed different, because selecting different features and feature engineering may increase performance. Last but not least, different Machine Learning models could be experimented with, as different models may generate a higher accuracy. 



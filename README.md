# Forest Cover Type Prediction: Use cartographic variables to classify forest categories

## Goal and Motivation 
The goal of this competition is to predict the type of forest cover using data given by the US Forest Service.  

## Cleaning Data
The data given for this competition contains no NA.  The only operation I performed is to condense the one-hot arrays for wilderness and soil types to integer arrays.

## Feature Engineering
The engineered features in this dataset can be found online on the Kaggle forum.
*Highwater - Check if the nearest water source is above or below the desinated area.
*Distance_To_Hydrology - Find the distance to the nearest water source.
*Hydro_Fire_1 -  Sum of distance to nearest water and wildfire ignition points
*Hydro_Fire_2 -  Absolute value of the difference of the distance to nearest water and wildfire ignition points.
*Hydro_Road_1 -  Absolute value of the sum of the distance to nearest water and roadway
*Hydro_Road_2 -  Absolute value of the difference of the distance to nearest water and roadway
*Fire_Road_1 -  Absolute value of the sum of the distance to nearest wildfire ignition points and roadway
*Fire_Road_2 -  Absolute value of the difference of the distance to nearest wildfire ignition points and roadway

## Choosing Classifiers
There are five classifiers used here: random forest, extra trees, support vector machine, k nearest neighbors, and logistic regression.  The optimal parameters are obtained by using RandomizedSearchCV with 10 folds cross validation.  The random forest and extra trees classifier both produced appromixately 80% accuracy, which are significantly better than the 75% produced by SVM, 71% by KNN, and 60% by logistic regression.  

## Result
Although random forest and extra trees produced similar cross validation accuracy, extra tress performed better on the test set: random forest produced 78% accuracy while extra trees produced 80%.  This puts me on the Kaggle ranking of the top 10% of the competitors if the competition is still active.


# Forest Cover Type Prediction: Use cartographic variables to classify forest categories

## Goal and Motivation 
The goal of this competition is to predict the type of forest cover using data given by the US Forest Service.  

## Cleaning Data
The data given for this competition contains no NA.  The only operation I performed is to condense the one-hot arrays for wilderness and soil types to integer arrays.

## Feature Engineering
The engineered features in this dataset can be found online on the Kaggle forum.
* Highwater - Check if the nearest water source is above or below the desinated area.
* Distance_To_Hydrology - Find the distance to the nearest water source.
* Hydro_Fire_1 -  Sum of distance to nearest water and wildfire ignition points.
* Hydro_Fire_2 -  Absolute value of the difference of the distance to nearest water and wildfire ignition points.
* Hydro_Road_1 -  Absolute value of the sum of the distance to nearest water and roadway.
* Hydro_Road_2 -  Absolute value of the difference of the distance to nearest water and roadway.
* Fire_Road_1 -  Absolute value of the sum of the distance to nearest wildfire ignition points and roadway.
* Fire_Road_2 -  Absolute value of the difference of the distance to nearest wildfire ignition points and roadway.

## Choosing Classifiers
The following five classifiers were chosen:
* Random Forest (RF)
* Extra Trees (ET)
* Support Vector Machine (SVM)
* K Nearest Neighbors
* Logistic Regression (LR)

The optimal parameters are obtained by using RandomizedSearchCV with 10 folds cross validation.  The scores produced are listed as followed:
* RF ~ 0.80
* ET ~ 0.80
* SVM ~ 0.75
* KNN ~ 0.71
* LR ~ 0.60

## Result
Although random forest and extra trees produced similar cross validation accuracy, extra tress performed better on the test set: random forest produced 78% accuracy while extra trees produced 80%.  This puts me on the Kaggle ranking of the top 10% of the competitors if the competition is still active.


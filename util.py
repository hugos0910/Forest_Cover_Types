import numpy as np
import pandas as pd
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def one_hot_to_integer(df):
  column_names = list(range(1,len(df.columns) + 1))
  df.columns = [column_names]
  col = df.idxmax(1)
  return col

def get_sanitized_data(df):
  # Condense wilderness type
  wilderness_type = df.ix[:,'Wilderness_Area1':'Wilderness_Area4']
  wilderness_integer = one_hot_to_integer(wilderness_type)

  # Condense soil type
  soil_type = df.ix[:,'Soil_Type1':'Soil_Type40']
  soil_integer = one_hot_to_integer(soil_type)

  df = df.drop(df.columns[11:], axis = 1)
  df = df.drop('Id', axis = 1)
  df['Wilderness'] = wilderness_integer
  df['Soil'] = soil_integer

  # Feature Engineering
  df['Highwater'] = df['Vertical_Distance_To_Hydrology'] < 0
  df['Distance_To_Hydrology'] = (df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)**0.5
  df['Hydro_Fire_1'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']
  df['Hydro_Fire_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])
  df['Hydro_Road_1'] = abs(df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways'])
  df['Hydro_Road_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])
  df['Fire_Road_1'] = abs(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Roadways'])
  df['Fire_Road_2'] = abs(df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways'])
  return df

def obtain_parameters(classifier_name, X_train, y, num_processor, num_iter):
  if classifier_name == 'RF':
    classifier = RandomForestClassifier()
    param_grid = dict(max_depth = [10, 20, 30, None], min_samples_split = [2,4,6], n_estimators = [500])
  elif classifier_name == 'ET':
    classifier = ExtraTreesClassifier()
    param_grid = dict(criterion = ['gini', 'entropy'],
                      max_depth = [10, 20, None], 
                      min_samples_split = [2,4,6], 
                      min_samples_leaf = [1,2,3], 
                      n_estimators = [500])
  elif classifier_name == 'SVM':
    steps = [('scl', StandardScaler()), 
             ('clf', SVC())]
    classifier = Pipeline(steps)
    param_grid = dict(clf__C = [1,5,10,15,20,25,30], 
                      clf__kernel = ['rbf'], 
                      clf__gamma = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1])
  elif classifier_name == 'KNN':
    steps = [('scl', StandardScaler()), 
             ('clf', KNeighborsClassifier())]
    classifier = Pipeline(steps)
    param_grid = dict(clf__n_neighbors = list(range(1,31)))
  elif classifier_name == 'LR':
    steps = [('scl', StandardScaler()), 
         ('clf', LogisticRegression())]
    classifier = Pipeline(steps)
    param_grid = dict(clf__penalty = ['l1', 'l2'], clf__C = [0.1,1,5,6,7,8,10,15,20,25,30])
  grid = RandomizedSearchCV(classifier, param_grid, cv = 10, scoring = 'accuracy', n_iter = num_iter, n_jobs = num_processor)
  
  grid.fit(X_train,y)
  grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
  print(grid_mean_scores)
  print(grid.best_estimator_)
  print(grid.best_params_)
  print('The best CV score using %s classifier is %f.' %(classifier_name, grid.best_score_))

def pretty_confusion_matrix(true, pred):
  y_actu = pd.Series(true, name='Actual')
  y_pred = pd.Series(pred, name='Predicted')
  df_confusion = pd.crosstab(y_actu, y_pred)
  percent = pd.Series(np.diag(df_confusion.values)/np.sum(df_confusion.values, axis = 1), name = 'Percentage')
  percent.index += 1
  df_confusion['Percentage'] = percent
  return df_confusion

import util
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Import data
print('Importing data...')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

'''
Choose features, available features are:

Original Features
  Elevation:                                Elevation in meters
  Aspect:                                   Aspect in degrees azimuth
  Slope:                                    Slope in degrees
  Horizontal_Distance_To_Hydrology:         Horz Dist to nearest surface water features
  Vertical_Distance_To_Hydrology:   
          Vert Dist to nearest surface water features
  Horizontal_Distance_To_Roadway:           Horz Dist to nearest roadway
  Hillshade_9am:                            Hillshade index at 9am, summer solstice
  Hillshade_Noon:                           Hillshade index at noon, summer solstice
  Hillshade_3pm:                            Hillshade index at 3pm, summer solsticee
  Horizontal_Distance_To_Fire_Points:       Horz Dist to nearest wildfire ignition points
  Wilderness_Area:                          Wilderness area designation
  Soil_Type:                                Soil Type designation
  Cover_Type:                               Forest Cover Type designation

Generated Features:
  Highwater:                                
  Distance_To_Hydrology:                    Total distance to nearest water source
  Hydro_Fire_1:                             Sum of distance to water and distance to fire points
  Hydro_Fire_2:                             ABS of differece of distance to water and distance to fire points
  Hydro_Road_1:                             ABS of sum of distance to water and distance to fire points
  Hydro_Road_2:                             ABS of differece of distance to water and distance to fire points
  Fire_Road_1:                              ABS of sum of distance to fire points and distance to roadway
  Fire_Road_2:                              ABS of differece of distance to fire points and distance to roadway
'''

print('Sanitizing data...')
train_clean = util.get_sanitized_data(train)
test_clean = util.get_sanitized_data(test)

features = train_clean.columns.tolist()

X_train = train_clean[features]
y = train.Cover_Type
X_test = test_clean[features]
test_id= test['Id']

# # Finding the best parameters
# classifier_name = 'RF'
# num_processor = 3
# n_iter = 10
# print('Finding the best parameters for %s classifier...' % classifier_name)
# util.obtain_parameters(classifier_name, X_train, y, num_processor, n_iter)

# Break test data into batches
batch = dict()
batch_size = 100000
start = 0
end = batch_size
for i in list(range(1, int(np.ceil(len(X_test)/batch_size))+1)):
  if i == int(np.ceil(len(X_test)/batch_size)):
    batch[i] = X_test.iloc[start:]
  batch[i] = X_test.iloc[start:end]
  start += batch_size
  end += batch_size
  
# Training classifier
classifier_name = 'ET'
print('Training %s classifier...' %classifier_name)
classifier = ExtraTreesClassifier(min_samples_split = 2, n_estimators = 500, max_depth = None, n_jobs = -1, random_state = 0)
classifier.fit(X_train, y)

# Making predictions
pred = dict()  
for i in list(range(1,7)):
  print('Predicting batch number %d' %i)
  if i == 1:
    prediction = classifier.predict(batch[i])
  else:
    prediction = np.concatenate((prediction, classifier.predict(batch[i])))
  
# Export data
print('Writing predictions to csv file...')
df_prediction = pd.DataFrame(prediction, index = test_id, columns = ["Cover_Type"])
df_prediction_csv = df_prediction.to_csv('prediction_%s.csv' %classifier_name, index_label = ["Id"])
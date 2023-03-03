# install packages
# pip install pycaret

from pycaret.datasets import get_data
dataset = get_data('employee')

# split test and train data

data_seen = dataset.sample(frac=0.95, random_state=780).reset_index(drop=True)
data_unseen = dataset.drop(data_seen.index).reset_index(drop=True)

dataset = dataset.drop(['department','average_montly_hours'], axis=1)
print('Data for modeling: '+str(data_seen.shape))
print('Data for prediction: '+str(data_unseen.shape))

#Upon running this, we get a tabular column of all the properties of our dataset 
#such as presence of missing values, whether PCA or other transformations are required, outliers etc.
import numpy
from pycaret.classification import *
setting_up = setup(data=data_seen, target='left', session_id=123)

compare_models()

# random forest seems to work the best (from above)
# use this model for dataset and train the seen and unseen
rf = create_model('rf')

# tuning hyperparameter automatically
tuned_model = tune_model(rf)

# finalise training process and predict the result
final = finalize_model(tuned_model)
unseen_predictions = predict_model(final, data=data_unseen)
unseen_predictions.head()

save_model(final, 'Final_model')
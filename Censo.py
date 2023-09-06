# import the relevant packages
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
## Preprocessing the data


# load the dataset
train_data = pd.read_csv("Census_income_train.csv")


# inspect the dataset
train_data.head()

# No null or NaN values
train_data.isnull().sum()

# All missing or unknow values, however, are marked with a question mark (?)
# There are 3 columns which contain '?' - Workclass, Occupation, Native-country

# We can obtain a list of boolean values indicating whether there is a '?' on the current row
train_data["Workclass"].str.contains("\?")

# Take the subset of the dataframe rows which don't contain '?'
clean_train_data = train_data[train_data["Workclass"].str.contains("\?") == False]

# Let's do the same for 'Occupation'
clean_train_data = clean_train_data[clean_train_data["Occupation"].str.contains("\?") == False]

# And for 'Native-country'
clean_train_data = clean_train_data[clean_train_data["Native-country"].str.contains("\?") == False]

# Finally, let's reset the index
clean_train_data = clean_train_data.reset_index(drop=True)

# In the original data, there are both categorical and numerical data
# Decision trees and random forest can work with categorical data in general
# However, this is not implemented in sklearn
# So, we need to convert the categorical data to numerical
# We will do that with one hot encoding

# Pandas can automatically do that for us with '.get_dummies'
train_dummies = pd.get_dummies(clean_train_data, drop_first=False)

# The last 2 columns are whether the income <= 50k and whether it is >50k
# Both of these carry the same information, so we will remove one of them
train_dummies = train_dummies.drop(['Income_ <=50K'],axis=1)

# The input features are everything besides the last column
train_input = train_dummies.iloc[:,:-1]

# The target/output is just the last column
train_target = train_dummies.iloc[:,-1]


## Let's do the same preprocessing on the test dataset
# Load test data
test_data = pd.read_csv("Census_income_test.csv")

clean_test_data = test_data[test_data["Workclass"].str.contains("\?") == False]

clean_test_data = clean_test_data[clean_test_data["Occupation"].str.contains("\?") == False]

clean_test_data = clean_test_data[clean_test_data["Native-country"].str.contains("\?") == False]

clean_test_data = clean_test_data.reset_index(drop=True)

test_dummies = pd.get_dummies(clean_test_data, drop_first=False)

test_dummies = test_dummies.drop(['Income_ <=50K.'],axis=1)

test_input = test_dummies.iloc[:,:-1]
test_target = test_dummies.iloc[:,-1]

# Initialize the model as a random forest classifier
clf = RandomForestClassifier()

# Train the model
clf.fit(train_input,train_target)

# Obtain the model's predictions on the test dataset
test_pred = clf.predict(test_input)

# Print the metrics obtained from the real targets and our model's predictions
print(classification_report(test_target, test_pred))

# Initialize the model as a random forest classifier with 150 trees (default is 100 trees)
clf = RandomForestClassifier(n_estimators = 150)


# Train the model
clf.fit(train_input,train_target)

# Obtain the model's predictions on the test dataset
test_pred = clf.predict(test_input)

# Print the metrics obtained from the real targets and our model's predictions
print(classification_report(test_target, test_pred))

# The result is basically the same as before, so the additional trees didn't help at all

##Creating and training the model

# Initialize the model as a random forest classifier with pruning
clf = RandomForestClassifier(ccp_alpha = 0.0001)

# Train the model
clf.fit(train_input,train_target)

# Obtain the model's predictions on the test dataset
test_pred = clf.predict(test_input)

# Print the metrics obtained from the real targets and our model's predictions
print(classification_report(test_target, test_pred))

# A slight increase in accuracy however it is insignificant
# This is the limit of the performance on this dataset
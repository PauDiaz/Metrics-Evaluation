#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

# Split data into training and testing
from sklearn import cross_validation  
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

# Fit data with sklearn decision trees algorithm
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

#get acuracy
from sklearn.metrics import accuracy_score
prediction = clf.predict(features_test)
#borrar=[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0. , 0.,  0.,  0.,  0.,  0.,  0.]


print accuracy_score(prediction, labels_test)
#print accuracy_score(borrar, labels_test)


# Precision and recall can help illuminate your performance better. Use the precision_score and recall_score
# available in sklearn.metrics to compute those quantities.

from sklearn.metrics import precision_score

print "Precision: ",precision_score(labels_test, prediction)


from sklearn.metrics import recall_score
print "Recall: ",recall_score(labels_test, prediction)

from sklearn.metrics import f1_score
print "F1_SCORE: ", f1_score(labels_test,prediction)

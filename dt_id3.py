import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn import tree
import os

df = pd.read_csv("breast_cancer_data.csv")
df = df.drop('id', axis = 1)

d = {'B' : False, 'M' : True}
df['diagnosis'] = df['diagnosis'].map(d)
df['diagnosis'] = df['diagnosis'].astype('int64')

X, y = df.drop('diagnosis', axis = 1), df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
cv_accuracies_by_k, test_accuracies_by_k = [], []

K = 4

'''
    Using K value for min samples split pruning method while building the DT
    
Higher min_samples_split values prevent a model from learning relations which might be highly specific to the particular
sample selected for a tree. Too high values can also lead to under-fitting hence depending on the level
of underfitting or overfitting.

'''

min_samples_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, min_samples_split = K)
val_scores = cross_val_score(estimator = min_samples_tree, X = X_train, y = y_train, cv = skf)
cv_accuracies_by_k.append(val_scores.mean())

min_samples_tree.fit(X_train, y_train)
curr_test_pred = min_samples_tree.predict(X_test)

test_accuracies_by_k.append(accuracy_score(y_test, curr_test_pred))
pred_test = min_samples_tree.predict(X_test)

print("(K Min-Samples Pruning)\nAccuracy: ", accuracy_score(y_test, pred_test))

export_graphviz(decision_tree = min_samples_tree,
               out_file='tree1.dot', filled=True, 
                feature_names=df.drop('diagnosis', axis=1).columns)

cv_accuracies_by_k, test_accuracies_by_k = [], []

'''
    Using K value for max depth pruning method while building the DT
    
the deeper we allow the tree to grow, the more complex the model will become because
we will have more splits and it captures more information about the data and this is one of the
root causes of overfitting in decision trees because the model will fit perfectly for the training data
and will not be able to generalize well on test set. So, if the model is overfitting, reducing the number
for max_depth is one way to combat overfitting.

'''

max_depth_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth = K)

val_scores = cross_val_score(estimator = max_depth_tree, X = X_train, y = y_train, cv = skf)
cv_accuracies_by_k.append(val_scores.mean())

max_depth_tree.fit(X_train, y_train)
curr_test_pred = max_depth_tree.predict(X_test)

test_accuracies_by_k.append(accuracy_score(y_test, curr_test_pred))

pred_test = max_depth_tree.predict(X_test)

print("\n(K Max-Depth Pruning)\nAccuracy: ", accuracy_score(y_test, pred_test))

export_graphviz(decision_tree = max_depth_tree,
               out_file='tree2.dot', filled=True, 
                feature_names=df.drop('diagnosis', axis=1).columns)

os.system('dot -Tpng tree1.dot -o tree1.png && dot -Tpng tree2.dot -o tree2.png')
os.system('open tree1.png && open tree2.png')
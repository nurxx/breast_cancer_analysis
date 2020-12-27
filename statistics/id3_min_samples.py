import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn import tree
import os 

df = pd.read_csv("breast_cancer_data.csv")
df = df.drop('id', axis = 1)

d = {'B' : False, 'M' : True}
df['diagnosis'] = df['diagnosis'].map(d)
df['diagnosis'] = df['diagnosis'].astype('int64')

X, y = df.drop('diagnosis', axis = 1), df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

tree = DecisionTreeClassifier(random_state = 0)
tree.fit(X_train, y_train)
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

cv_accuracies_by_k, test_accuracies_by_k = [], []

min_samples_split = np.arange(2, 5)

for k in tqdm(min_samples_split):
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, min_samples_split = k)
    
    val_scores = cross_val_score(estimator = tree, X = X_train, y = y_train, cv = skf)
    cv_accuracies_by_k.append(val_scores.mean())

    tree.fit(X_train, y_train)
    curr_test_pred = tree.predict(X_test)
    
    test_accuracies_by_k.append(accuracy_score(y_test, curr_test_pred))

plt.plot(min_samples_split, cv_accuracies_by_k, label = 'Cross-validation', c = 'blue')
plt.plot(min_samples_split, test_accuracies_by_k, label = 'Test', c = 'orange')
plt.legend()
plt.xlabel('K (min-samples)')
plt.ylabel('Accuracy')
plt.title('Decision Tree validation curves for K')

pred_test = tree.predict(X_test)

print("(K Min-Samples Pruning)\nAccuracy: ", accuracy_score(y_test, pred_test))

plt.show()
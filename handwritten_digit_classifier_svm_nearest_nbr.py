import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")


train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")

train.columns.values[0] = "label"
#print train

#to see the bias of dataset towrads any digit
sns.countplot(train['label'])
plt.show()

#print train['label'].value_counts()

features = train.columns[1:]
X = train[features]
y = train['label']
X_train, X_test, y_train, y_test = train_test_split(X/255.,y,test_size=0.1,random_state=0)



start_time=time.time()
clf = neighbors.KNeighborsClassifier(n_neighbors=5,p=2,weights='distance')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
end_time=time.time()


print "confusion matrix KNearest Nbr: \n ", confusion_matrix(y_test, y_pred)


for i in (np.random.randint(0,270,6)):
 two_d = (np.reshape(X_test.values[i], (28, 28)) * 255).astype(np.uint8)
 plt.title('predicted label: {0}'. format(y_pred[i]))
 plt.imshow(two_d, interpolation='nearest', cmap='gray')
 plt.show()


acc_knn = accuracy_score(y_test, y_pred)
print "nearest neighbors accuracy: ",acc_knn
print("Test set predict time: {}s.".format(end_time-start_time))


clf_svm = LinearSVC()
clf_svm.fit(X_train, y_train)
y_pred = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred)
print "Linear SVM accuracy: ",acc_svm

print "confusion matrix SVM: \n ", confusion_matrix(y_test, y_pred)

for i in (np.random.randint(0,270,6)):
 two_d = (np.reshape(X_test.values[i], (28, 28)) * 255).astype(np.uint8)
 plt.title('predicted label: {0}'. format(y_pred[i]))
 plt.imshow(two_d, interpolation='nearest', cmap='gray')
 plt.show()













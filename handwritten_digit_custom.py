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

train = pd.read_csv("MNIST_test_small_include_hand_new1.csv")
test = pd.read_csv("MNIST_test_small_include_hand_new1.csv")

column_list = list(test.columns.values)
print ("length of column list",len(column_list))


train.columns.values[0] = "label"
#print train


new_file_df_hand = pd.read_csv('MNIST_test_small_include_hand_new1.csv')
X_hand_train = new_file_df_hand.iloc[0:9000, 1:]
X_hand_test  = new_file_df_hand.iloc[9000:, 1:]
y_hand_test = new_file_df_hand.iloc[9000:, 0]
y_hand_train = new_file_df_hand.iloc[0:9000, 0]

print("printing last ten values")
print new_file_df_hand.tail(10)

print("printing lengths..................")
print(len(X_hand_train))
print(len(X_hand_test))

start_time=time.time()
clf = neighbors.KNeighborsClassifier(n_neighbors=5,p=2,weights='distance')
clf.fit(X_hand_train,y_hand_train)
y_hand_pred = clf.predict(X_hand_test)
end_time=time.time()

acc_knn = accuracy_score(y_hand_test, y_hand_pred)
print "nearest neighbors accuracy: ",acc_knn
print("Test set predict time: {}s.".format(end_time-start_time))


print("printing Nearest Neighbor classifier handwritten values")

print("printing predicted and test values")
print(len(y_hand_test))
print(len(y_hand_pred))
length = len(y_hand_test)


print type(y_hand_pred), len(y_hand_pred)
print "predicted list: ", y_hand_pred[995:]
print("\n")
print "real handwritten list\n", y_hand_test[995:]

for ik in range(995, 1000, 1):
 three_d = (np.reshape(X_hand_test.values[ik], (28, 28)) * 255).astype(np.uint8)
 plt.title('predicted label: {0}'. format(y_hand_pred[ik]))
 plt.imshow(three_d, interpolation='nearest', cmap='gray')
 plt.show()



clf_svm = LinearSVC()
clf_svm.fit(X_hand_train, y_hand_train)
y_hand_pred = clf_svm.predict(X_hand_test)
acc_svm = accuracy_score(y_hand_test, y_hand_pred)
print "Linear SVM accuracy: ",acc_svm



print("printing predicted and test values")
print(len(y_hand_test))
print(len(y_hand_pred))
length = len(y_hand_test)


print type(y_hand_pred), len(y_hand_pred)
print "predicted list: ", y_hand_pred[995:]
print("\n")
print "real handwritten list\n", y_hand_test[995:]

for ik in range(995, 1000, 1):
 three_d = (np.reshape(X_hand_test.values[ik], (28, 28)) * 255).astype(np.uint8)
 plt.title('predicted label: {0}'. format(y_hand_pred[ik]))
 plt.imshow(three_d, interpolation='nearest', cmap='gray')
 plt.show()



#####################################################################################
####################################################################################
#          SVM Start

#X_tr = train.iloc[:,1:]
#y_tr = train.iloc[:, 0]

#X_train, X_test, y_train, y_test = train_test_split(X_tr,y_tr,test_size=0.1, random_state=0, stratify=y_tr)

#steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='poly'))]
#pipeline = Pipeline(steps)

#parameters = {'SVM__C':[0.001, 0.1, 100, 10e5], 'SVM__gamma':[10,1,0.1,0.01]}
#grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)

#grid.fit(X_train, y_train)
#print "score = %3.2f" %(grid.score(X_test, y_test))

#print "best parameters from train data: ", grid.best_params_






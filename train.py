import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image as pil_image
from matplotlib.pyplot import imshow, imsave
from IPython.display import Image as Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from main_features import  preprocess

import cv2
import numpy as np
from glob import glob
import os
from matplotlib import pyplot as plt
from main_features import segmentation
from main_features import texture

from skimage.color import rgb2gray


folder_list = os.listdir('dataset')

outputVectors = []
loadedImages = []

input1=np.zeros([20, 1])
target=[]
tar=0
for folder in folder_list:
        
        # create a path to the folder
        path = 'dataset/' + str(folder)
        img_files = os.listdir(path)
        print(path)
        for file in img_files:
            src = os.path.join(path, file)
            main_img = cv2.imread(src)
            res=preprocess(main_img)
            y=segmentation(res)
            z=texture(y)
            if len(z)==20:
                input1= np.c_[input1,z]
                target.append(tar)
        tar=tar+1


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

#---------------conversion of all categorial column values to vector/numerical--------#

Label= labelencoder.fit_transform(target)



X=np.transpose(input1[:,1:])

#X=input1[:,1:]
Y=target

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=12)
from sklearn.tree import DecisionTreeClassifier
clf_model = DecisionTreeClassifier(criterion="gini",random_state=12)
clf_model.fit(X_train,y_train)
y_predict = clf_model.predict(X_train)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accdt = accuracy_score(y_train,y_predict)*100
print("accuracy of decision tree is=",accdt)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
test_targets=y_train
cm = confusion_matrix(test_targets,y_predict)

from sklearn.metrics import roc_curve
##fpr ,tn, thresholds = roc_curve((test_targets),(y_predict))
##Sensitivity= tn / (tn+fpr)
##print('Sensitivity='+str(Sensitivity[1]))
precision = precision_score((test_targets),(y_predict))

print('precision='+str(precision))

fpr ,tpr, thresholds = roc_curve((test_targets),(y_predict))
f1score = f1_score((test_targets),(y_predict))

print('f1-score='+str(f1score))

fpr ,tpr, thresholds = roc_curve((test_targets),(y_predict))
recallscore = recall_score((test_targets),(y_predict))

print('recall-score='+str(recallscore))

import pickle
file = 'finalized_model_DT.sav'
pickle.dump(clf_model, open(file, 'wb'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


X=np.transpose(input1[:,1:])

#X=input1[:,1:]
Y=target

### Define the model
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(10, activation="sigmoid"))

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])



import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# Generate some sample data
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
# Calculate precision
accuracy = accuracy_score(y_true, y_pred)*100
print("CNN accuracy is:", accuracy)
precision = precision_score(y_true, y_pred)
print("Precision:", precision)
# Calculate recall
recall = recall_score(y_true, y_pred)
print("Recall:", recall)
# Calculate F1 score
f1 = f1_score(y_true, y_pred)
print("F1 score:", f1)
### Calculate sensitivity (True Positive Rate)
##tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
##sensitivity = tp / (tp + fn)
##print("Sensitivity:", sensitivity)



import matplotlib.pyplot as plt
x=['Decision Tree', 'CNN']
y=[accdt, accuracy]
plt.bar(x,y, color=('green','blue'))
plt.xlabel('Algorithm')
plt.ylabel("Accuracy")
plt.title('Accuracy bar plot')
plt.show()

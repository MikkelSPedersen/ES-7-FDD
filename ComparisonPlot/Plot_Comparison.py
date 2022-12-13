import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.axes as ax
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import glob
import natsort # sort the file names numerically
from numpy import asarray

trainLabel = np.ravel(pd.read_csv(r".\trainingLabel.csv"))
valLabel =  np.ravel(pd.read_csv(r".\valLabel.csv"))

plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=18)
#############
#   BDT-Full-Train
#############
labelPredict = np.ravel(pd.read_csv(r".\BDT_Full_Train.csv"))
accuracy = round(accuracy_score(trainLabel, labelPredict),2)
cm = confusion_matrix(trainLabel, labelPredict,normalize='true')
disp = ConfusionMatrixDisplay(np.around(cm, 2), display_labels=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]))
disp.plot(cmap='Blues')
disp.ax_.set(title='BDT Full Dataset Train \n Accuracy: ' + str(accuracy))
#############
#   BDT-Full-Validation 
#############
labelPredict = np.ravel(pd.read_csv(r".\BDT_Full_Val.csv"))
accuracy = round(accuracy_score(valLabel, labelPredict),2)
cm = confusion_matrix(valLabel, labelPredict,normalize='true')
disp = ConfusionMatrixDisplay(np.around(cm, 2), display_labels=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]))
disp.plot(cmap='Blues')
disp.ax_.set(title='BDT Full Dataset Validation \n Accuracy: ' + str(accuracy))
#############
#   BDT-LDA-Train
#############
labelPredict = np.ravel(pd.read_csv(r".\BDT_LDA_Train.csv"))
accuracy = round(accuracy_score(trainLabel, labelPredict),2)
cm = confusion_matrix(trainLabel, labelPredict,normalize='true')
disp = ConfusionMatrixDisplay(np.around(cm, 2), display_labels=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]))
disp.plot(cmap='Blues')
disp.ax_.set(title='BDT LDA Dataset Train \n Accuracy: ' + str(accuracy))
#############
#   BDT-LDA-Validation 
#############
labelPredict = np.ravel(pd.read_csv(r".\BDT_LDA_Val.csv"))
accuracy = round(accuracy_score(valLabel, labelPredict),2)
cm = confusion_matrix(valLabel, labelPredict,normalize='true')
disp = ConfusionMatrixDisplay(np.around(cm, 2), display_labels=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]))
disp.plot(cmap='Blues')
disp.ax_.set(title='BDT LDA Dataset Validation \n Accuracy: ' + str(accuracy))
#############
#   BDT-PCA-Train
#############
labelPredict = np.ravel(pd.read_csv(r".\BDT_PCA_Train.csv"))
accuracy = round(accuracy_score(trainLabel, labelPredict),2)
cm = confusion_matrix(trainLabel, labelPredict,normalize='true')
disp = ConfusionMatrixDisplay(np.around(cm, 2), display_labels=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]))
disp.plot(cmap='Blues')
disp.ax_.set(title='BDT PCA Dataset Train \n Accuracy: ' + str(accuracy))
#############
#   BDT-PCA-Validation
#############
labelPredict = np.ravel(pd.read_csv(r".\BDT_PCA_Val.csv"))
accuracy = round(accuracy_score(valLabel, labelPredict),2)
cm = confusion_matrix(valLabel, labelPredict,normalize='true')
disp = ConfusionMatrixDisplay(np.around(cm, 2), display_labels=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]))
disp.plot(cmap='Blues')
disp.ax_.set(title='BDT PCA Dataset Validation \n Accuracy: ' + str(accuracy))


#############
#   SVM-Full-Train
#############
labelPredict = np.ravel(pd.read_csv(r".\SVM_Full_Train.csv"))
accuracy = round(accuracy_score(trainLabel, labelPredict),2)
cm = confusion_matrix(trainLabel, labelPredict,normalize='true')
disp = ConfusionMatrixDisplay(np.around(cm, 2), display_labels=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]))
disp.plot(cmap='Blues')
disp.ax_.set(title='SVM Full Dataset Train \n Accuracy: ' + str(accuracy))
#############
#   SVM-Full-Validation 
#############
labelPredict = np.ravel(pd.read_csv(r".\SVM_Full_Val.csv"))
accuracy = round(accuracy_score(valLabel, labelPredict),2)
cm = confusion_matrix(valLabel, labelPredict,normalize='true')
disp = ConfusionMatrixDisplay(np.around(cm, 2), display_labels=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]))
disp.plot(cmap='Blues')
disp.ax_.set(title='SVM Full Dataset Validation \n Accuracy: ' + str(accuracy))
#############
#   SVM-LDA-Train
#############
labelPredict = np.ravel(pd.read_csv(r".\SVM_LDA_Train.csv"))
accuracy = round(accuracy_score(trainLabel, labelPredict),2)
cm = confusion_matrix(trainLabel, labelPredict,normalize='true')
disp = ConfusionMatrixDisplay(np.around(cm, 2), display_labels=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]))
disp.plot(cmap='Blues')
disp.ax_.set(title='SVM LDA Dataset Train \n Accuracy: ' + str(accuracy))
#############
#   SVM-LDA-Validation 
#############
labelPredict = np.ravel(pd.read_csv(r".\SVM_LDA_Val.csv"))
accuracy = round(accuracy_score(valLabel, labelPredict),2)
cm = confusion_matrix(valLabel, labelPredict,normalize='true')
disp = ConfusionMatrixDisplay(np.around(cm, 2), display_labels=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]))
disp.plot(cmap='Blues')
disp.ax_.set(title='SVM LDA Dataset Validation \n Accuracy: ' + str(accuracy))
#############
#   SVM-PCA-Train
#############
labelPredict = np.ravel(pd.read_csv(r".\SVM_PCA_Train.csv"))
accuracy = round(accuracy_score(trainLabel, labelPredict),2)
cm = confusion_matrix(trainLabel, labelPredict,normalize='true')
disp = ConfusionMatrixDisplay(np.around(cm, 2), display_labels=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]))
disp.plot(cmap='Blues')
disp.ax_.set(title='SVM PCA Dataset Train \n Accuracy: ' + str(accuracy))
#############
#   SVM-PCA-Validation
#############
labelPredict = np.ravel(pd.read_csv(r".\SVM_PCA_Val.csv"))
accuracy = round(accuracy_score(valLabel, labelPredict),2)
cm = confusion_matrix(valLabel, labelPredict,normalize='true')
disp = ConfusionMatrixDisplay(np.around(cm, 2), display_labels=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]))
disp.plot(cmap='Blues')
disp.ax_.set(title='SVM PCA Dataset Validation \n Accuracy: ' + str(accuracy))


plt.show()

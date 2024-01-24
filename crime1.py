from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from imutils import paths
from tkinter.filedialog import askopenfilename

import numpy as np 
import pandas as pd 


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

import pylab as pl
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
pl.style.use('fivethirtyeight')
import seaborn as sns
from os import path

import warnings
warnings.filterwarnings('ignore')

main = tkinter.Tk()
main.title("Crime Prediction and Analysis Using Machine Learning")
main.geometry("1300x1200")

class test:
	def upload():
            global filename
            text.delete('1.0', END)
            filename = askopenfilename(initialdir = ".")
            pathlabel.config(text=filename)
            text.insert(END,"Dataset loaded\n\n")

	def csv():
            global X,y
            global data
            text.delete('1.0', END)
            data=pd.read_csv(filename)
            text.insert(END,"Top Five rows of dataset\n"+str(data.head())+"\n")
            crimedata = data.drop(["Date", "ID", "Case_Number", "Block", "IUCR", "Description", "X_Coordinate", "Y_Coordinate", "Year", "Updated_On", "Latitude", "Longitude", "Location"], axis=1)
            crimedata = crimedata.replace('NaN', 0)
            crimeInfo = pd.get_dummies(crimedata)
            X = crimeInfo.drop("Arrest", axis=1)
            y = crimeInfo["Arrest"]
            text.insert(END, "Shape of X : "+str(X.shape)+"\n Shape of Y: "+str(y.shape)+"\n")
            X = StandardScaler().fit_transform(X)
		
	def splitdataset():		 
            text.delete('1.0', END)
            global X,y
            global X_train,X_test,y_train, y_test
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1 ,stratify=y)
            text.insert(END,"\nTrain & Test Model Generated\n\n")
            text.insert(END,"Total Dataset Size : "+str(len(data))+"\n")
            text.insert(END,"Split Training Size : "+str(len(X_train))+"\n")
            text.insert(END,"Split Test Size : "+str(len(X_test))+"\n")
	def logit():		 
            global X_train,X_test,y_train, y_test
            global lgmodel,lgacc
            text.delete('1.0', END)
            
            if (path.exists('models/lgmodel.sav') == False):
                lgmodel = LogisticRegression()
                lgmodel.fit(X_train, y_train)
                pickle.dump(lgmodel, open('models/lgmodel.sav', 'wb'))
            else:
                lgmodel = pickle.load(open('models/lgmodel.sav', 'rb'))

            lgacc = lgmodel.score(X_test,y_test)
            text.insert(END, "Logistic Training Data Score: "+str(lgmodel.score(X_train, y_train)*100)+"\n")
            text.insert(END, "Logistic Testing Data Score: "+str(lgmodel.score(X_test, y_test)*100)+"\n")
            
            predictions = lgmodel.predict(X_test)
            print(f"Logistic First 10 Predictions:   {predictions[:10]}")
            print(f"Logistic First 10 Actual labels: {y_test[:10].tolist()}")


	def knn():		 
            global X_train,X_test,y_train, y_test
            global knnmodel,knnacc
            
            knnmodel = KNeighborsClassifier(n_neighbors=9)
            knnmodel.fit(X_train, y_train) 
            pickle.dump(knnmodel, open('knnmodel.sav', 'wb'))
            
            knnacc = knnmodel.score(X_test,y_test)
            text.insert(END, "KNN Training Data Score: "+str(knnmodel.score(X_train, y_train)*100)+"\n")
            text.insert(END, "KNN Testing Data Score: "+str(knnmodel.score(X_test, y_test)*100)+"\n")
            
            predictions = knnmodel.predict(X_test)
            print(f"KNN First 10 Predictions:   {predictions[:10]}")
            print(f"KNN First 10 Actual labels: {y_test[:10].tolist()}")
	def dt():		 
            global X_train,X_test,y_train, y_test
            global dtmodel,dtacc
            
            if (path.exists('models/dtmodel.sav') == False):
                dtmodel = DecisionTreeClassifier()
                dtmodel.fit(X_train, y_train) 
                pickle.dump(dtmodel, open('models/dtmodel.sav', 'wb'))
            else:
                dtmodel = pickle.load(open('models/dtmodel.sav', 'rb'))
            
            dtacc = dtmodel.score(X_test,y_test)
            text.insert(END, "Decision Tree Training Data Score: "+str(dtmodel.score(X_train, y_train)*100)+"\n")
            text.insert(END, "Decision Tree Testing Data Score: "+str(dtmodel.score(X_test, y_test)*100)+"\n")
            
            predictions = dtmodel.predict(X_test)
            print(f"DT First 10 Predictions:   {predictions[:10]}")
            print(f"DT First 10 Actual labels: {y_test[:10].tolist()}")
	
	def nb():		 
            global X_train,X_test,y_train, y_test
            global nbmodel,nbacc
            
            if (path.exists('models/dtmodel.sav') == False):
                nbmodel = GaussianNB()
                nbmodel.fit(X_train, y_train) 
                pickle.dump(nbmodel, open('models/nbmodel.sav', 'wb'))
            else:
                nbmodel = pickle.load(open('models/nbmodel.sav', 'rb'))
            
            nbacc = nbmodel.score(X_test,y_test)
            text.insert(END, "Naive Bayes Training Data Score: "+str(nbmodel.score(X_train, y_train)*100)+"\n")
            text.insert(END, "Naive Bayes Testing Data Score: "+str(nbmodel.score(X_test, y_test)*100)+"\n")
            
            predictions = nbmodel.predict(X_test)
            print(f"NB First 10 Predictions:   {predictions[:10]}")
            print(f"NB First 10 Actual labels: {y_test[:10].tolist()}")
	
	def rf():		 
            global X_train,X_test,y_train, y_test
            global rfmodel,rfacc
            
            if (path.exists('models/rfmodel.sav') == False):
                rfmodel = RandomForestClassifier(n_estimators=200)
                rfmodel.fit(X_train, y_train) 
                pickle.dump(rfmodel, open('models/rfmodel.sav', 'wb'))
            else:
                rfmodel = pickle.load(open('models/rfmodel.sav', 'rb'))
            
            rfacc = rfmodel.score(X_test,y_test)
            text.insert(END, "RandomForest Training Data Score: "+str(rfmodel.score(X_train, y_train)*100)+"\n")
            text.insert(END, "RandomForest Testing Data Score: "+str(rfmodel.score(X_test, y_test)*100)+"\n")
            
            predictions = rfmodel.predict(X_test)
            print(f"RF First 10 Predictions:   {predictions[:10]}")
            print(f"RF First 10 Actual labels: {y_test[:10].tolist()}")
        
	def svc():		 
            global X_train,X_test,y_train, y_test
            global svcmodel,svcacc
            
            if (path.exists('models/svcmodel.sav') == False):
                svcmodel = SVC(gamma='auto')
                svcmodel.fit(X_train, y_train) 
                pickle.dump(svcmodel, open('models/svcmodel.sav', 'wb'))
            else:
                svcmodel = pickle.load(open('models/svcmodel.sav', 'rb'))
            
            svcacc = svcmodel.score(X_test,y_test)
            text.insert(END, "SVM Training Data Score: "+str(svcmodel.score(X_train, y_train)*100)+"\n")
            text.insert(END, "SVM Testing Data Score: "+str(svcmodel.score(X_test, y_test)*100)+"\n")
            
            predictions = svcmodel.predict(X_test)
            print(f"SVC First 10 Predictions:   {predictions[:10]}")
            print(f"SVC First 10 Actual labels: {y_test[:10].tolist()}")

	def graph():		 
	    results=[knnacc,lgacc,dtacc,nbacc,svcacc,rfacc]	    
	    bars = ('KNN', 'LOGIT', 'CART','NB','SVM','RF')
	    y_pos = np.arange(len(bars))
	    plt.bar(y_pos, results)
	    plt.xticks(y_pos, bars)
	    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Crime Prediction Using Machine Learning')
title.config(bg='sky blue', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset", command=test.upload)
upload.place(x=700,y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='dark orchid', fg='white')  
pathlabel.config(font=font1)
pathlabel.place(x=700,y=150)

df = Button(main, text="Reading Data ", command=test.csv)
df.place(x=700,y=200)
df.config(font=font1)

split = Button(main, text="Train_Test_Split ", command=test.splitdataset)
split.place(x=700,y=250)
split.config(font=font1)

lg= Button(main, text="Logistic Classifier", command=test.logit)
lg.place(x=700,y=300)
lg.config(font=font1) 
"""
knn= Button(main, text="KNN Classifier", command=test.knn)
knn.place(x=700,y=350)
knn.config(font=font1) 

"""

nb = Button(main, text="NB Classifier", command=test.nb)
nb.place(x=700,y=350)
nb.config(font=font1)

dt = Button(main, text="DT Classifier", command=test.dt)
dt.place(x=700,y=400)
dt.config(font=font1)

rf = Button(main, text="RF Classifier", command=test.rf)
rf.place(x=700,y=450)
rf.config(font=font1)

svc = Button(main, text="SVC Classifier", command=test.svc)
svc.place(x=700,y=500)
svc.config(font=font1)

graph= Button(main, text="Model Comparison", command=test.graph)
graph.place(x=700,y=550)
graph.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='pale goldenrod')
main.mainloop()

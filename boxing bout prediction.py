# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 20:30:40 2018

@author: Christian.GANDON
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:50:48 2018

@author: antoinekrainc
"""

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# full execution with logs and graphs?
Full = 0


# import Dataset 
dataset = pd.read_csv("data\Boxing_bout_original.csv", sep=';')

# elimination des scores des juges et du type de victoire
# elimination des stands car systématiquement == (erreur dataset?)

# comment faire des ranges de colonnes pour ne pas les définir une par une??
d0 = dataset.iloc[:,[0,1,2,3,4,5,8,9,10,11,12,13,14,15,16,17,18]]
d0 = d0.dropna()

#encodage variable "victoire"

d0["A_won"] = 0
  
for i in d0.index:
    if d0.loc[i,"result"] == "win_A": 
        d0.loc[i,"A_won"] = 1
    else:
        d0.loc[i,"A_won"] = 0

#encodage des autres variables
d1 = d0.loc[:,"A_won"]

d1 = pd.DataFrame(d1)

d1["result"] = d0["result"]
d1["age_gap"] = d0["age_A"] - d0["age_B"]
d1["height_gap"] = d0["height_A"] - d0["height_B"]
d1["reach_gap"] = d0["reach_A"] - d0["reach_B"]
d1["weight_gap"] = d0["weight_A"] - d0["weight_B"]

# les criteres historiques sont ramenés a une variable 
d1["win_histo"] = d0["won_A"] - d0["lost_A"] - d0["won_B"] + d0["lost_B"] 

#les KO historiques sont estimatés en rapports au différentiel de victoires historiques
d1["kos_prop"] = ( d0["kos_A"] / d0["won_A"] ) - ( d0["kos_B"] / d0["won_B"] ) 



# élimination des valeurs abérrantes sur age and reach gaps
for i in d1.index:
    if (((d1.loc[i,"age_gap"])**2)**(1/2)) > 50:
        if Full == 1:
            print("éliminé: index ", i, d1.loc[i,"age_gap"], ((d1.loc[i,"age_gap"])**2)**(1/2))
        d1.loc[i,"age_gap"] = None
     
    if (((d1.loc[i,"reach_gap"])**2)**(1/2)) > 50:
        if Full == 1:
            print("éliminé: index ", i, d1.loc[i,"reach_gap"], ((d1.loc[i,"reach_gap"])**2)**(1/2))
        d1.loc[i,"reach_gap"] = None

d1 = d1.dropna()


# pour éviter la surpondération des "A_won" de 80%, j'append le data set une deuxieme fois en inversant tous les paramètres et le résultat


dbis = pd.DataFrame()

dbis["A_won"] = ((d1["A_won"] - 1)**2)**(1/2)
dbis["result"] = d1["result"]
dbis["age_gap"] = d1["age_gap"] * -1
dbis["height_gap"] = d1["height_gap"] * -1
dbis["reach_gap"] = d1["reach_gap"] * -1
dbis["weight_gap"] = d1["weight_gap"] * -1
dbis["win_histo"] =  d1["win_histo"] * -1
dbis["kos_prop"] = d1["kos_prop"] * -1

    
    
d1 = d1.append(dbis)


# exploration
if Full == 1:
    sns.pairplot(d1, hue="result")
    
    sns.lmplot(x="age_gap", y="A_won", data = d1, logistic = True)
    sns.lmplot(x="height_gap", y="A_won", data = d1, logistic = True)
    sns.lmplot(x="reach_gap", y="A_won", data = d1, logistic = True)
    sns.lmplot(x="weight_gap", y="A_won", data = d1, logistic = True)
    sns.lmplot(x="win_histo", y="A_won", data = d1, logistic = True)
    sns.lmplot(x="kos_prop", y="A_won", data = d1, logistic = True)
# hypothèse 

X = d1.iloc[:,2:9]
Y = d1.loc[:,"A_won"]

# Divide dataset Train set & Test set 
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 10000)

# Feature Scaling <=> normalization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Application du modèle de classification
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

# Decision Tree Classifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

print(model)

# make predictions
expected = Y_test
predicted = model.predict(X_test)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

"""
# prédire
Y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# CM matrix très mauvais, essayer un kbest pour mieux sélectionner les features

#recherche du modele optimum
import statsmodels.formula.api as sm
# méthode des moindres carré (ordinary least square)
classifier_logit = sm.Logit(endog = Y, exog = X).fit()
classifier_logit .summary()

# nouveau test sans le weight
X_opt = X_train[:,[0,1,2,4,5]]
classifier_logit = sm.Logit(endog = Y_train, exog = X_opt).fit()
classifier_logit .summary()

# nouveau test sans le weight et sans le reach
X_opt = X_train[:,[0,1,4,5]]
classifier_logit = sm.Logit(endog = Y_train, exog = X_opt).fit()
classifier_logit .summary()

# nouveau cycle avec modèle optimisé

classifier.fit(X_opt, Y_train)

X_test_opt = X_test[:,[0,1,4,5]]
Y_pred_opt = classifier.predict(X_test_opt)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm_opt = confusion_matrix(Y_test, Y_pred_opt)
"""

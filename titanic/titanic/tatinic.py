# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

for dataset in combine:
    dataset = dataset.drop(['Cabin'], axis=1)
    print(dataset.info())
    # dataset['Embarked'] = dataset['Embarked'].fillna('S')
    # dataset['Age'] = dataset['Age'].fillna(0)
    # dataset['Fare'] = dataset['Fare'].fillna(0)
    # #print(dataset.info())

print(train_df.info())

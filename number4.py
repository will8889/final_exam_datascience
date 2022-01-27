import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

encoder = LabelEncoder()

df=pd.read_csv('./creditscoring.csv')
df.info()
df.head(20)

df.drop(['ID','DAYS_BIRTH','AMT_INCOME_TOTAL'], axis=1, inplace=True)
list = ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE']
for x in list:
    df[x] = encoder.fit_transform(df[x])

df.info()
y = df.pop('TARGET')
x = df

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

model = svm.SVC(kernel='linear')
model.fit(x_train,y_train)

print('prediction :',model.predict(x_test)[:10])
print('actual :',y_test[:10].values)

scores = cross_val_score(model, x_train, y_train, cv=10)
print(scores)
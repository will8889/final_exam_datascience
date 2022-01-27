import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
encoder = LabelEncoder()

df=pd.read_csv('./creditscoring.csv')
df.info()
df.head(20)

df.drop(['ID','CNT_CHILDREN','DAYS_BIRTH','FLAG_OWN_REALTY','FLAG_OWN_CAR','NAME_FAMILY_STATUS','FLAG_MOBIL','FLAG_WORK_PHONE','FLAG_PHONE','FLAG_EMAIL','CNT_FAM_MEMBERS','BEGIN_MONTH'], axis=1, inplace=True)
df['OCCUPATION_TYPE'] = encoder.fit_transform(df['OCCUPATION_TYPE'])
df['NAME_INCOME_TYPE'] = encoder.fit_transform(df['NAME_INCOME_TYPE'])
df['NAME_EDUCATION_TYPE'] = encoder.fit_transform(df['NAME_EDUCATION_TYPE'])
df['NAME_HOUSING_TYPE'] = encoder.fit_transform(df['NAME_HOUSING_TYPE'])
df['CODE_GENDER'] = encoder.fit_transform(df['CODE_GENDER'])
df1 = df.pop('AMT_INCOME_TOTAL')
df['AMT_INCOME_TOTAL']=df1
df.info()

y = df.pop('AMT_INCOME_TOTAL')
x = df

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)

ols = LinearRegression()
ols.fit(x_train,y_train)

print('r2 score',ols.score(x_test,y_test))
print('adjusted r2 score =',1-(1- ols.score(x_test,y_test)) * (len(y_test)-1)/(len(y_test)-x_test.shape[1]-1))

predict = ols.predict(x_test)
print('mean absolute error', mean_absolute_error(y_test,predict))

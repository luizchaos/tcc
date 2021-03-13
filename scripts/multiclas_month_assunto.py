import pandas as pd

print("reading data")
df = pd.read_csv('dados_ouvidoria.csv',sep=';',error_bad_lines=False,encoding='utf-8')


print("cleaning")
df['DATA_PERGUNTA'] = pd.to_datetime(df['DATA_PERGUNTA'], errors='coerce')

df.dropna()

df['MM'] = (df['DATA_PERGUNTA'].dt.month)

df['DD'] = (df['DATA_PERGUNTA'].dt.day)

df_18 = df[df.DATA_PERGUNTA.dt.year == 2018]

df_17 = df[df.DATA_PERGUNTA.dt.year == 2017]

# df_16 = df[df.DATA_PERGUNTA.dt.year == 2016]

frames = [df_17, df_18]

df_train = pd.concat(frames)


df_test = df[df.DATA_PERGUNTA.dt.year == 2019 ]



print("axis")
# X_tr= df_train.DATA_PERGUNTA
X_tr= df_train.MM
# X_tr= df_train.drop(axis=0, columns=['MM', 'DD'])#.astype(float)#df_train.MM_DD
Y_tr = df_train.TIPO

# X_te = df_test.DATA_PERGUNTA
X_te = df_test.MM
# X_te = df_test.drop(axis=0, columns=['MM', 'DD'])#.astype(float)#df_train.MM_DD
Y_te = df_test.TIPO

# X = df.DATA_PERGUNTA
# Y = df.ASSUNTO


print("test/train")
from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, Y,  test_size=0.33, random_state=42)

# X_train= X_train.values.reshape(-1, 1)
# y_train= y_train.values.reshape(-1, 1)
# X_test = X_test.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)

X_train= X_tr.values.reshape(-1, 1)
# X_train= X_tr
y_train= Y_tr.values.reshape(-1, 1)
X_test = X_te.values.reshape(-1, 1)
# X_test = X_te
y_test = Y_te.values.reshape(-1, 1)


print("training model")
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier

import numpy as np

extra_tree = ExtraTreeClassifier(random_state=0)
cls = BaggingClassifier(extra_tree, random_state=0).fit( X_train, np.ravel(y_train,order='C'))

print("trained")

print(cls.score(X_test, y_test))

print("graph")

# import matplotlib.pyplot as plt

predictions = cls.predict_proba(X_test)

import plotly.express as px
import plotly.graph_objects as go

x_range = np.linspace(X_train.min(), X_train.max(), 100)
y_range = cls.predict(x_range.reshape(-1, 1))

fig = px.scatter(df, x='MM', y='TIPO', opacity=0.65)
fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
fig.show()


# plt.plot(np.ravel(y_test,order='C'))
# plt.plot(predictions)

# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# df_original = pd.DataFrame({X_test, y_test})

# df_model = pd.DataFrame({X_test, predictions})


# plt.scatter( np.ravel(y_test,order='C'), X_test)
# plt.scatter( predictions)

# plt.show()
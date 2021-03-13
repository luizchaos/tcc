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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

extra_tree = ExtraTreeClassifier(random_state=0)
bag = BaggingClassifier(extra_tree, random_state=0).fit( X_train, np.ravel(y_train,order='C'))

dec_tr = DecisionTreeClassifier(random_state=0)
dec_tr.fit(X_train, np.ravel(y_train,order='C'))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, np.ravel(y_train,order='C'))

rf = RandomForestClassifier(max_depth=2, random_state=0)
rf.fit(X_train, np.ravel(y_train,order='C'))

print("trained")

# print(cls.score(X_test, y_test))

print("graph")

# import matplotlib.pyplot as plt

# predictions_bag = bag.predict_proba(X_test)
# predictions_dec_tr = dec_tr.predict_proba(X_test)
# predictions_knn = knn.predict_proba(X_test)
# predictions_rf = rf.predict_proba(X_test)

# print("bag")
# print(predictions_bag)

# print("dec_tr")
# print(predictions_dec_tr)

# probas = model.predict_proba(dataframe)

classes = bag.classes_

bag_array = []
for class_name in classes:
    class_index = np.where(bag.classes_ == class_name)
    proba = bag.predict_proba(X_test)
    obj = {
        "class": class_name,
        "proba": proba
    }
    bag_array.append(obj)
    # print(f"{class_name}: {proba}")



print("bag")
print(bag_array)


# import matplotlib.pyplot as plt
# plt.plot(predictions_bag)
# plt.show()
# import plotly.express as px
# import plotly.graph_objects as go

# x_range = np.linspace(X_train.min(), X_train.max(), 100)
# y_range_bag = bag.predict(x_range.reshape(-1, 1))
# y_range_dec_tr = dec_tr.predict(x_range.reshape(-1, 1))
# y_range_knn = knn.predict(x_range.reshape(-1, 1))
# y_range_rf = rf.predict(x_range.reshape(-1, 1))

# fig = px.scatter(df_test, x='DATA_PERGUNTA', y='TIPO', opacity=0.65)
# fig.add_traces(go.Scatter(x=x_range, y=y_range_bag, name='Bagging'))
# fig.add_traces(go.Scatter(x=x_range, y=y_range_dec_tr, name='Decision Tree'))
# fig.add_traces(go.Scatter(x=x_range, y=y_range_knn, name='KNN'))
# fig.add_traces(go.Scatter(x=x_range, y=y_range_rf, name='Random Forest'))
# fig.show()



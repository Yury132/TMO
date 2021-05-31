from sklearn.datasets import *
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

@st.cache
def load_data():
    cancer = load_breast_cancer()
    return cancer

def make_dataframe(ds_function):
    ds = ds_function()
    df = pd.DataFrame(data= np.c_[ds['data'], ds['target']],
                     columns= list(ds['feature_names']) + ['target'])
    return df


st.title('ИУ5-65Б Усынин Юрий Лаб №6')
st.header('Обучение модели ближайших соседей')

data_load_state = st.text('Загрузка данных...')

cancer = load_data()

data_load_state.text('Данные загружены!')

# Количество строчек
data_len = cancer.target.shape[0]

# Разделяем выборку
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=1)

# Формируем DataFrame
cancer_df = make_dataframe(load_breast_cancer)
st.subheader('Первые 5 значений')
st.write(cancer_df.head())

if st.checkbox('Показать статистические характеристики'):
    st.subheader('Данные')
    st.write(cancer_df.describe())

st.subheader('Размеры выборки:')
st.write('Размер обучающей выборки X - {}'.format(X_train.shape))
st.write('Размер тестовой выборки X - {}'.format(X_test.shape))
st.write('Размер обучающей выборки Y - {}'.format(y_train.shape))
st.write('Размер тестовой выборки Y - {}'.format(y_test.shape))

# Кол-во фолдов
folds = st.slider('Количество фолдов:', min_value=3, max_value=10, value=5, step=1)

# Количество строчек в одном фолде
rows_in_one_fold = int(data_len / folds)
# Количество возможных ближайших соседей
allowed_knn = int(rows_in_one_fold * (folds-1))
st.write('Количество строк в наборе данных - {}'.format(data_len))
st.write('Максимальное допустимое количество ближайших соседей с учетом выбранного количества фолдов - {}'.format(allowed_knn))

# Кол-во соседей
neighbors = st.slider('Количество ближайших соседей:', min_value=1, max_value=allowed_knn, value=5, step=1)


st.subheader('Обучение модели и оценка качества')
model = KNeighborsClassifier(n_neighbors=neighbors)
# Обучение модели
model.fit(X_train, y_train)
# Предсказание
target = model.predict(X_test)
# Метрики
st.subheader('Метрики качества:')
st.write('Accuracy = {}'.format(accuracy_score(y_test, target)))
st.write('F-мера = {}'.format(f1_score(y_test, target, average='macro')))


# Функция для отрисовки ROC-кривой
def draw_roc_curve(y_true, y_score, pos_label, average):
    fpr, tpr, thresholds = roc_curve(y_true, y_score,
                                     pos_label=pos_label)
    roc_auc_value = roc_auc_score(y_true, y_score, average=average)
    fig1 = plt.figure(figsize=(7,5))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_value)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #plt.show()
    st.pyplot(fig1)

# Рисуем ROC-кривую
draw_roc_curve(y_test, target, pos_label=1, average='micro')

# Оценка качества модели с использованием кросс-валидации
# KFold стратегия
scores = cross_val_score(KNeighborsClassifier(n_neighbors=neighbors),
    cancer.data, cancer.target, scoring='accuracy', cv=folds)

st.subheader('Оценка качества модели с использованием кросс-валидации')
st.write('Значения accuracy для отдельных фолдов')
st.bar_chart(scores)
st.write('Усредненное значение accuracy по всем фолдам - {}'.format(np.mean(scores)))
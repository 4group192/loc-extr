import streamlit as st
from Classes.sympy_extremum import Extremum, array_to_df
import time
from numpy import sin, cos, tan, exp, pi

def page1():
    st.title('Поиск экстремумов ФНП')
    st.write('Приложение ищет экстремумы функции двух переменных в действительной плоскости')
    st.header('Задания 1, 2')
    #Настройка условий
    func = st.text_input('Введите целевую функцию', value = 'y*(x**2)+x*(y**3) - x*y')
    task = st.selectbox(label = 'Выберите тип задачи', options = ['Безусловный экстремум', 'Условный экстремум'])
    if task == 'Условный экстремум':
        g = st.text_input('Введите ограничивающую функцию', value = 'x**2+4*y**2 - 1')
    x_d = st.number_input('Введите нижнюю границу для x', value = -10)
    x_up = st.number_input('Введите верхнюю границу для x', value = 10)
    y_d = st.number_input('Введите нижнюю границу для y ', value = -1)
    y_up = st.number_input('Введите верхнюю границу для y ', value= 1)
    limits = [[x_d, x_up], [y_d, y_up]]


    if task == 'Условный экстремум':
        """ После каждого нажатия следующей кнопки (дальше будут кнопки) переменные под блоком кнопки удалятся из памяти,
        поэтому мы определяем объект класса здесь, в дальнейшем он нам пригодится для третьего задания
        """
        Example = Extremum(func = func, g = g, limits = limits)
    else:
        Example = Extremum(func = func, limits = limits)

    if st.button('Найти экстремумы и построить график'):
        if task =='Условный экстремум':
            st.dataframe(Example.lagr())
            st.plotly_chart(Example.visualize())
        else:
            st.dataframe(Example.extremums())
            st.plotly_chart(Example.visualize())

    st.header('Задание 3')
    learn_rate = st.slider('Выберите learn_rate', min_value = 0.00, max_value = 1.00, step = 0.05)
    n_iter = st.slider('Выберите макс кол-во итераций', min_value = 0, max_value = 200)
    if st.button('Сравнить производительность разных алгоритмов'):
        st.dataframe(Example.time_compare(learn_rate, n_iter))
        st.write('Точка минимума/седловая точка, полученная градиентным спуском')
        st.dataframe(array_to_df(Example.gradient_descent(learn_rate= learn_rate, n_iter = n_iter)))
        st.write('Стац. точки, полученные аналитическим способом')
        st.dataframe(Example.extremums())

def page2():
    st.write('Не сделан')

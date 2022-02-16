import streamlit as st
from Classes.sympy_extremum import Extremum

st.title('Поиск экстремумов ФНП')
st.header('Задания 1, 2')
#Настройка условий
variables = st.text_input('Введите название переменных, например x y', value = 'x y')
func = st.text_input('Введите целевую функцию', value = 'y*(x**2)+x*(y**3) - x*y')
task = st.selectbox(label = 'Выберите тип задачи', options = ['Безусловный экстремум', 'Условный экстремум'])
if task == 'Условный экстремум':
    g = st.text_input('Введите ограничивающую функцию', value = 'x**2+4*y**2 - 1')
x_d = st.number_input('Введите нижнюю границу для x', value = -10)
x_up = st.number_input('Введите верхнюю границу для x', value = 10)
y_d = st.number_input('Введите нижнюю границу для y ', value = -1)
y_up = st.number_input('Введите верхнюю границу для y ', value= 1)
limits = [[x_d, x_up], [y_d, y_up]]

if st.button('Найти экстремумы и построить график'):
    if task =='Условный экстремум':
        try:
            Example = Extremum(variables=variables, func=lambda x, y: eval(func), g=lambda x, y: eval(g), limits=limits)
        except AttributeError:
            if limits[0][0] > limits[0][1]:
                st.text('Oops!  Not correct   limit input.  Try again...')
            if limits[1][0] > limits[1][1]:
                st.text('Oops!  Not correct y  limit input.  Try again...')
            else:
                st.text('Oops!  Not correct variables input.  Try again...')
        else:
            st.dataframe(Example.lagr())
            st.plotly_chart(Example.visualize())
    else:
        try:
            Example = Extremum(variables=variables, func=lambda x, y: eval(func), limits=limits)
        except AttributeError:
            if limits[0][0] > limits[0][1]:
                st.text('Oops!  Not correct x  limit input.  Try again...')
            if limits[1][0] > limits[1][1]:
                st.text('Oops!  Not correct y  limit input.  Try again...')
            else:
                st.text('Oops!  Not correct variables input.  Try again...')
        else:
            st.dataframe(Example.extremums())
            st.plotly_chart(Example.visualize())


import streamlit as st
from page1 import *
from Classes.sympy_extremum import Extremum

st.sidebar.header('Выберите проект')

task = st.sidebar.selectbox('Choose', ['Поиск экстремумов ФНП', '2-й проект'])

task_dict = {
    'Поиск экстремумов ФНП': page1
}
if st.sidebar.button('Click'):
    task_dict[task]()
=======
if st.button('Найти экстремумы и построить график'):
    if task =='Условный экстремум':
        try:
            Example = Extremum(variables=variables, func=lambda x, y: eval(func), g=lambda x, y: eval(g), limits=limits)
        except AttributeError:
            if limits[0][0] > limits[0][1]:
                st.text('Oops!  Not correct   limit input.  Try again...')
            if limits[1][0] > limits[1][1]:
                st.text('Oops!  Not correct y  limit input.  Try again...')
        except TypeError:
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
        except TypeError:
            st.text('Oops!  Not correct variables input.  Try again...')
        else:
            st.dataframe(Example.extremums())
            st.plotly_chart(Example.visualize())


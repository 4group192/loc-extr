import streamlit as st
from Classes import *
from Classes.sympy_extremum import Extremum

st.title('Поиск экстремумов ФНП')
Example1 = Extremum('x y', lambda x, y: y*(x**2)+x*(y**3) - x*y, limits=[[-10, 10], [-1, 1]])
st.write('Найденные экстремумы:')
st.write(Example1.extremums())
st.plotly_chart(Example1.visualize2())
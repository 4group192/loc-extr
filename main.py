import streamlit as st
from page1 import *

st.sidebar.header('Выберите проект')

task = st.sidebar.selectbox('Choose', ['Поиск экстремумов ФНП', '2-й проект'])

task_dict = {
    'Поиск экстремумов ФНП': page1
}
if st.sidebar.button('Click'):
    task_dict[task]()
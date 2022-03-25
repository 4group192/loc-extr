import streamlit as st
from Classes.streamlit_code import *
project = st.sidebar.selectbox("Выберите проект", options = [
    'Поиск экстремумов ФНП', 
    'Методы одномерной оптимизации',
    'Градиентные методы многомерной оптимизации'])
    
projects = {'Поиск экстремумов ФНП': page1,
            'Методы одномерной оптимизации': page2,
            'Градиентные методы многомерной оптимизации': page3
}
projects[project]()
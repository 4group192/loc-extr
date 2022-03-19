import streamlit as st
from pages import *
project = st.sidebar.selectbox("Выберите проект", options = ['Поиск экстремумов ФНП', 'Методы одномерной оптимизации'])
projects = {'Поиск экстремумов ФНП': page1,
            'Методы одномерной оптимизации': page2
}
projects[project]()
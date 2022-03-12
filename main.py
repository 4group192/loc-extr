import streamlit as st
from pages import *
project = st.sidebar.selectbox("Выберите проект", options = ['Поиск экстремумов ФНП', '2-й (не сделан)'])
projects = {'Поиск экстремумов ФНП': page1,
            '2-й (не сделан)': page2
}
projects[project]()
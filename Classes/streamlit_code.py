import streamlit as st
from Classes import fnp, mop, gradient_methods
import time
from numpy import sin, cos, tan, exp, pi

def page1():
    st.title('Поиск экстремумов ФНП')
    st.write('Приложение ищет экстремумы функции двух переменных в действительной плоскости')
    #Настройка условий
    func = st.sidebar.text_input('Введите целевую функцию', value = 'y*(x**2)+x*(y**3) - x*y')
    task = st.sidebar.selectbox(label = 'Выберите тип задачи', options = ['Безусловный экстремум', 'Условный экстремум'])
    if task == 'Условный экстремум':
        g = st.sidebar.text_input('Введите ограничивающую функцию', value = 'x**2+4*y**2 - 1')
    x_d = st.sidebar.number_input('Введите нижнюю границу для x', value = -10)
    x_up = st.sidebar.number_input('Введите верхнюю границу для x', value = 10)
    y_d = st.sidebar.number_input('Введите нижнюю границу для y ', value = -1)
    y_up = st.sidebar.number_input('Введите верхнюю границу для y ', value= 1)
    limits = [[x_d, x_up], [y_d, y_up]]


    if task == 'Условный экстремум':
        """ После каждого нажатия следующей кнопки (дальше будут кнопки) переменные под блоком кнопки удалятся из памяти,
        поэтому мы определяем объект класса здесь, в дальнейшем он нам пригодится для третьего задания
        """
        Example = fnp.Extremum(func = func, g = g, limits = limits)
    else:
        Example = fnp.Extremum(func = func, limits = limits)

    if st.sidebar.button('Найти экстремумы и построить график'):
        st.header('Задания 1, 2')
        if task =='Условный экстремум':
            st.dataframe(Example.lagr())
            st.plotly_chart(Example.visualize())
        else:
            st.dataframe(Example.extremums())
            st.plotly_chart(Example.visualize())

    learn_rate = st.sidebar.slider('Выберите learn_rate', min_value = 0.00, max_value = 1.00, step = 0.05)
    n_iter = st.sidebar.slider('Выберите макс кол-во итераций', min_value = 0, max_value = 200)
    if st.sidebar.button('Сравнить производительность разных алгоритмов'):
        st.header('Задание 3')
        st.dataframe(Example.time_compare(learn_rate, n_iter))
        st.write('Точка минимума/седловая точка, полученная градиентным спуском')
        st.dataframe(fnp.array_to_df(Example.gradient_descent(learn_rate= learn_rate, n_iter = n_iter)))
        st.write('Стац. точки, полученные аналитическим способом')
        st.dataframe(Example.extremums())

def page2():
    st.title('Поиск экстремумов функции одной перемонной')
    st.write('Приложение ищет экстремумы функции одной переменной в действительной плоскости')
    #Настройка условий
    func = st.sidebar.text_input('Введите целевую функцию', value = 'x**2 - 2*x + 10')
    task = st.sidebar.selectbox(label = 'Выберите метод', options = ['Метод золотого сечения', 'Метод парабол', 'Метод Брента', 'Алгоритм Бройдена — Флетчера — Гольдфарба — Шанно'])
    a = st.sidebar.number_input('Введите нижнюю границу интервала', value =-0.5)
    b = st.sidebar.number_input('Введите верхнюю границу интервала', value = 0.5)
    eps = st.sidebar.number_input('Введите точность', value = 0.00005)
    max_iter = st.sidebar.number_input('Введите макс. кол-во итераций', value = 500)
    PIR = st.sidebar.selectbox(label = 'Вывод промежуточных результатов', options=[False, True])
    SIR = st.sidebar.selectbox(label = 'Сохранение промежуточных результатов', options=[False, True])
    st.sidebar.write('Параметры для BFGS')
    x0 = st.sidebar.number_input('Введите начальную точку', value =1)
    max_x = st.sidebar.number_input('Введите максимальное значение аргумента функции', value =100)
    c1 = st.sidebar.number_input('Введите параметр для первого условия Вольфе', value =10**(-4))
    c2 = st.sidebar.number_input('Введите параметр для второго условия Вольфе', value =0.1)
    Example = mop.Extremum_1d(func =func, a = a, b = b, eps = eps, max_iter = max_iter, print_intermediate_results= PIR, save_intermediate_results= SIR, x0=x0, max_x=max_x,c1=c1,c2=c2)
    Extra_Tasks = mop.ExtraTasks(func =func, a = a, b = b, eps = eps, max_iter = max_iter, print_intermediate_results= False, save_intermediate_results= True, method=task, x0=x0, max_x=max_x,c1=c1,c2=c2)
    funcs = {
        'Метод золотого сечения': Example.gss,
        'Метод парабол': Example.quadratic_approximation,
        'Метод Брента': Example.brent,
        'Алгоритм Бройдена — Флетчера — Гольдфарба — Шанно': Example.BFGS}

    if st.sidebar.button('Click'):
        st.header('Задание 1. ' + task)
        st.dataframe(funcs[task]().iloc[:,[0,1,3]])

        st.header('Задание 3. График работы методы')
        st.plotly_chart(Extra_Tasks.q3())

        st.header('Задание 4. График сходимости алгоритма')
        st.plotly_chart(Extra_Tasks.q4())

        st.header('Задание 5. Сравнение производительности алгоритмов')
        st.dataframe(Extra_Tasks.q5())

def page3():
    st.title('Многомерная (до 4-х переменных) оптимизация с помощью градиентных методов')
    st.sidebar.header('Ввод данных')
    func = st.sidebar.text_input('Целевая функция', value = 'x1**2 - 2*x1*x2 + x2**2')
    method = st.sidebar.selectbox('Выберите метод', options=[
        'Градиентный спуск с постоянным шагом',
        'Градиентный спуск с дроблением шага',
        'Метод наискорейшего спуска',
        'Метод сопряженных градиентов'])
    x0 = st.sidebar.text_input('Список координат начальной точки. Пример для 3-х переменных: [2, 5, 1]')
    max_iterations = st.sidebar.slider('Макс. кол-во итераций, ', min_value=1, max_value=1000, value=500, step=25)
    if method != 'Метод сопряженных градиентов':
        lr = st.sidebar.slider('learning rate', min_value=0.05, max_value=1.0, value=0.2, step=0.05)
        PIR = st.sidebar.selectbox(label = 'Вывод промежуточных результатов', options=[False, True])
        SIR = st.sidebar.selectbox(label = 'Сохранение промежуточных результатов', options=[False, True])
    #func, n_variables, x0, max_iterations, eps, SIR, PIR, lr

    if st.sidebar.button('Найти минимум'):
        if method != 'Метод сопряженных градиентов':
            result = gradient_methods.GradientMethod(func = func, x0 = x0, max_iterations = max_iterations, SIR = SIR, PIR = PIR, lr = lr).minimize2(method)
        else:
            result = gradient_methods.GradientMethod(func = func, x0 = x0, max_iterations = max_iterations).minimize2(method)
        st.write(result)
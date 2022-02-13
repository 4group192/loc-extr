from sympy import *
import numpy as np
import timeit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

class Extremum:
    """
    Класс для поиска и визуализации экстремумов

    Attributes
    ----------
    x, y: np.array
        Массив X, Y. Пределы этих массивов устанавливает пользователь (limits), по умолчанию пределы: [-1000, 1000]
    X, Y: np.array
        Массивы координатных сеток 2-мерного координатного пространства для массивов X и Y. (Преобразованные массивы для работы
        с 2-мерной функцией)
    z: np.array
        Массив значений целевой функции
    symb_1, symb_2: Symbol
        Введенные пользователем символы для работы с библиотекой sympy.
    analytic_func: sympy.core.function.Function
        Функция, введенная пользователем, в виде sympy.Function.
    
    Methods
    -------
    extremums(self)
        Находит все критические точки и определяет их типы
    visualize(self)
        Строит график целевой функции и наносит на него найденные критические точки
    """
    def __init__(self, variables: str, func, limits=None):
        """
        Parametres
        ----------
        variables: str
            Названия переменных
        func: lambda
            Целевая функция
        limits: list
            Ограничения для переменных
        """
        if limits is None:
            self.x = np.arange(-1000, 1000, 0.1)
            self.y = np.arange(-1000, 1000, 0.1)
        else:
            if limits[0] is not None:
                assert limits[0][0] < limits[0][1]
                self.x = np.arange(limits[0][0], limits[0][1], 0.1)
                
            if limits[1] is not None:
                assert limits[1][0] < limits[1][1]
                self.y = np.arange(limits[1][0], limits[1][1], 0.1)

        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.z = func(self.X, self.Y)
        self.symb_1, self.symb_2 = symbols(variables)
        self.analytic_func = func(self.symb_1, self.symb_2)

    def extremums(self):
        """Находит все критические точки и опредляет их типы

        Returns:
            pandas.Dataframe
        """
        diff_x = self.analytic_func.diff(self.symb_1)
        diff_y = self.analytic_func.diff(self.symb_2)
        critical_values = solve([diff_x, diff_y], [self.symb_1, self.symb_2])

        diff_xx = diff_x.diff(self.symb_1)
        diff_xy = diff_x.diff(self.symb_2)
        diff_yy = diff_y.diff(self.symb_2)

        self.df_points = pd.DataFrame(columns=['x', 'y', 'val', 'type'])
        func_silv = diff_xx * diff_yy - diff_xy**2
        for point in critical_values:
            A = diff_xx.subs(self.symb_1, point[0]).subs(self.symb_2, point[1])
            val = func_silv.subs(self.symb_1, point[0]).subs(self.symb_2, point[1])
            row = {'x': float(point[0]), 'y': float(point[1]), 'val': float(val)}
            if val == 0:
                row['type'] = 'Требуется доп. исследование'
            elif val < 0:
                row['type'] = 'Седловая точка'
            else:
                if A > 0:
                    row['type'] = 'Min'
                else:
                    row['type'] = 'Max'
            self.df_points = self.df_points.append(row, ignore_index = True)
        return self.df_points

    def visualize(self):
        """Строит график целевой функции и наносит на него найденные критические точки
        """
        fig = go.Figure(data=[go.Surface(z=self.z, x=self.x, y=self.y)])
        for  point_type in set(self.df_points.type):
            z = self.df_points[self.df_points.type == point_type].val
            x = self.df_points[self.df_points.type == point_type].x
            y = self.df_points[self.df_points.type == point_type].y
            fig.add_trace(go.Scatter3d(z = z, x = x, y= y, name = point_type, showlegend = True))
        fig.update_layout(title=str(self.analytic_func), autosize=True,
                  width=1000, height=1000,
                  margin=dict(l=65, r=50, b=65, t=90))
        return fig




if __name__ == '__main__': 
    Example1 = Extremum('x y', lambda x, y: y*(x**2)+x*(y**3) - x*y, limits=[[-10, 10], [-10, 10]])
    print(Example1.extremums())
    Example1.visualize().show()
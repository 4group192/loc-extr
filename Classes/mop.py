# Методы одномерной оптимизации

import numpy as np
import time
import pandas as pd
from streamlit import write
from sympy import dict_merge
import plotly.graph_objects as go

class Extremum_1d:
    def __init__(self, 
                func: str,
                a,
                b,
                eps = 10**(-5),
                max_iter = 500,
                print_intermediate_results = False,
                save_intermediate_results = False):
        self.func = lambda x: eval(func)
        self.a = a
        self.b = b
        self.eps = eps
        self.max_iter = max_iter
        self.PIR = print_intermediate_results
        self.SIR = save_intermediate_results
        self.results = None


    def gss(self):
        a,b = self.a, self.b
        gr = (np.sqrt(5) + 1) / 2
        f = self.func
        self.results = pd.DataFrame(columns=['x', 'f(x)', 'Величина исследуемого интервала', 'Отчет о работе алгоритма'])

        c = b - (b - a) / gr
        d = a + (b - a) / gr
        n_iter = 0
        while abs(b - a)/2 > self.eps and n_iter < self.max_iter:
            if f(c) < f(d):
                b = d
            else:
                a = c

            # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            n_iter += 1
            if self.PIR:
                write(f'x = {(a + b) / 2}, f(x) = {f((a + b) / 2)}, iter = {n_iter}')
            if self.SIR and n_iter < self.max_iter and abs(a - b)/2 > self.eps:
               self.results = self.results.append({'x': (a+b)/2, 'f(x)': f((a+b)/2), 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 3}, ignore_index=True)

        if n_iter == self.max_iter:
            self.results = self.results.append({'x': (a+b)/2, 'f(x)': f((a+b)/2), 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 1}, ignore_index=True)
        else:
            self.results = self.results.append({'x': (a+b)/2, 'f(x)': f((a+b)/2), 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 0}, ignore_index=True)
        return self.results

    def quadratic_approximation(self):
        try:
            func = self.func
            a, b, c = self.a, self.b, (self.a + self.b) / 2
            f0, f1, f2 = func(a), func(b), func(c)
            f_x = {a: f0, b: f1, c: f2}
            self.results = pd.DataFrame(columns=['x', 'f(x)', 'Величина исследуемого интервала', 'Отчет о работе алгоритма'])
            n_iter = 0
            while n_iter < self.max_iter and abs(b - c)/2 > self.eps:
                f0, f1, f2 = f_x[a], f_x[b], f_x[c]
                p = (b - c) ** 2 * (f2 - f0) + (a - c) ** 2 * (f1 - f2)
                q = 2 * ((b - c) * (f2 - f0) + (a - c) * (f1 - f2))
                assert p != 0
                assert q != 0

                x_new = c + p / q
                assert self.a <= x_new <= self.b

                f_new = func(x_new)
                f_x[x_new] = f_new
                previous_xs = [a, b, c]

                if f_new < f2:
                    a, f0 = b, f1
                    b, f1 = x_new, f_new
                    c, f2 = x_new, f_new

                elif f_new < f1:
                    a, f0 = b, f1
                    b, f1 = x_new, f_new

                elif f_new < f0:
                    a, f0 = x_new, f_new
                
                n_iter += 1

                if self.PIR:
                    write(f'x = {c}, f(x) = {f2}, iter = {n_iter}')

                if self.SIR and n_iter < self.max_iter and abs(b - c)/2 > self.eps:
                        self.results = self.results.append({'x': c, 'f(x)': f2, 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 3}, ignore_index=True)        
            
            if n_iter == self.max_iter:
                self.results = self.results.append({'x': c, 'f(x)': f2, 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 1}, ignore_index=True)
            else:
                self.results = self.results.append({'x': c, 'f(x)': f2, 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 0}, ignore_index=True)
            return self.results
        except Exception as e:
            self.results = self.results.append({'x': c, 'f(x)': f2, 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 2}, ignore_index=True)
            return self.results
            
        
class ExtraTasks(Extremum_1d):
    def __init__(self, 
                func: str,
                a,
                b,
                eps = 10**(-5),
                max_iter = 500,
                print_intermediate_results = False,
                save_intermediate_results = True,
                method = 'Метод золотого сечения'):
        Extremum_1d.__init__(self, 
                func,
                a,
                b,
                eps,
                max_iter,
                print_intermediate_results,
                save_intermediate_results)
        self.dict_method = {
        'Метод золотого сечения': self.gss,
        'Метод парабол': self.quadratic_approximation,
        'Метод Брента': self.gss,
        'Алгоритм Бройдена — Флетчера — Гольдфарба — Шанно': self.gss}
        self.results = self.dict_method[method]()

    def q3(self):
        x = np.linspace(self.a, self.b)
        y = self.func(x)
        c = self.results['f(x)'].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y,
                    mode='lines',
                    name='f(x)'))
        for i in range(len(self.results)):
            fig.add_trace(go.Scatter(x=[self.results['x'][i]], y=[self.results['f(x)'][i]],
                    mode='markers',
                    name=f'iter {i}'))
        return fig

    def q4(self):
        x = self.results.index + 1
        y = self.results['Величина исследуемого интервала'].values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y,
                    mode='lines'))
        fig.update_layout(title='Сходимость алгоритма',
                   xaxis_title='Итерации',
                   yaxis_title='Величина исследуемого интервала')
        return fig

    def q5(self):
        df = pd.DataFrame(columns= ['Получено решение', 'Время выполнения', 'Кол-во итераций'], index = self.dict_method.keys())
        for method in df.index:
            start_time = time.time()
            result = self.dict_method[method]()
            exec_time = time.time() - start_time

            df.loc[method, 'Получено решение'] = f'x = {result.iloc[[-1], [0]].values[0][0]}, f(x) = {result.iloc[[-1], [1]].values[0][0]}'
            df.loc[method, 'Время выполнения'] = exec_time
            df.loc[method, 'Кол-во итераций'] =  len(result.index) - 1
        return df


            

if __name__ == '__main__':
    ExtraTasks('-5*x**5 + 4*x**4 - 12*x**3  + 11*x**2 - 2*x + 1',-0.5,0.5,print_intermediate_results=True, save_intermediate_results=True,method = 'Метод золотого сечения').visualize().show()
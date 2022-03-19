# Методы одномерной оптимизации

import numpy as np
import time
import pandas as pd
from streamlit import write

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
        self.results = pd.DataFrame(columns=['x', 'f(x)', 'Отчет о работе алгоритма'])

        c = b - (b - a) / gr
        d = a + (b - a) / gr
        iter = 0
        while abs(b - a)/2 > self.eps and iter < self.max_iter:
            if f(c) < f(d):
                b = d
            else:
                a = c

            # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            iter += 1
            if self.PIR:
                write(f'x = {(a + b) / 2}, f(x) = {f((a + b) / 2)}, iter = {iter}')
            if self.SIR:
                if abs(b - a)/2 <= self.eps:
                    self.results = self.results.append({'x': (a+b)/2, 'f(x)': f((a+b)/2), 'Отчет о работе алгоритма': 0}, ignore_index=True)
                elif iter == self.max_iter:
                    self.results = self.results.append({'x': (a+b)/2, 'f(x)': f((a+b)/2), 'Отчет о работе алгоритма': 1}, ignore_index=True)
                else:
                    self.results = self.results.append({'x': (a+b)/2, 'f(x)': f((a+b)/2), 'Отчет о работе алгоритма': 2}, ignore_index=True)

        return self.results

    def quadratic_approximation(self):
        func = self.func
        x0, x1, x2 = self.a, self.b, (self.a + self.b) / 2
        f0, f1, f2 = func(x0), func(x1), func(x2)
        f_x = {x0: f0, x1: f1, x2: f2}
        self.results = pd.DataFrame(columns=['x', 'f(x)', 'Отчет о работе алгоритма'])
        n_iter = 0
        while n_iter < self.max_iter and abs(x1 - x2)/2 > self.eps:
            f0, f1, f2 = f_x[x0], f_x[x1], f_x[x2]
            p = (x1 - x2) ** 2 * (f2 - f0) + (x0 - x2) ** 2 * (f1 - f2)
            q = 2 * ((x1 - x2) * (f2 - f0) + (x0 - x2) * (f1 - f2))
            assert p != 0
            assert q != 0

            x_new = x2 + p / q
            assert self.a <= x_new <= self.b

            f_new = func(x_new)
            f_x[x_new] = f_new
            previous_xs = [x0, x1, x2]

            if f_new < f2:
                x0, f0 = x1, f1
                x1, f1 = x_new, f_new
                x2, f2 = x_new, f_new

            elif f_new < f1:
                x0, f0 = x1, f1
                x1, f1 = x_new, f_new

            elif f_new < f0:
                x0, f0 = x_new, f_new
            
            n_iter += 1

            if self.PIR:
                write(f'x = {x2}, f(x) = {f2}, iter = {n_iter}')

            if self.SIR:
                if abs(x2 - x1)/2 <= self.eps:
                    self.results = self.results.append({'x': x2, 'f(x)': f2, 'Отчет о работе алгоритма': 0}, ignore_index=True)
                elif n_iter == self.max_iter:
                    self.results = self.results.append({'x': x2, 'f(x)': f2, 'Отчет о работе алгоритма': 1}, ignore_index=True)
                else:
                    self.results = self.results.append({'x': x2, 'f(x)': f2, 'Отчет о работе алгоритма': 2}, ignore_index=True)
            
        return self.results
            
        

            

if __name__ == '__main__':
    print(Extremum_1d('x ** 3 - x ** 2 - x',-3,10,print_intermediate_results=True, save_intermediate_results=True).quadratic_approximation())
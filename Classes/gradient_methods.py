#Градиентные методы

from typing import List
from xmlrpc.client import Boolean
import numpy as np
import scipy
from scipy.optimize import minimize_scalar, minimize, approx_fprime
import sympy
from numpy import *
import time
import pandas as pd
from streamlit import write
from sympy import dict_merge, solve
import plotly.graph_objects as go

class GradientMethod:
    def __init__(self, func, x0, max_iterations = 500, eps = 1e-5, SIR = False, PIR = False, lr = 0.2):
        self.func = lambda x1, x2, x3, x4: eval(func)
        self.n_variables = len(eval(x0))
        self.x0 = np.array([0, 0, 0, 0], dtype='float64')
        for i in range(self.n_variables):
            self.x0[i] = eval(x0)[i]
        
        self.max_iterations = max_iterations
        self.eps = eps
        self.SIR = SIR
        self.PIR = PIR
        self.lr = lr
        self.dataset = pd.DataFrame(columns=['X', 'f(X)'])
        self.dataset.index_name = 'Iteration'

    def function(self, x):
        x1, x2, x3, x4 = x
        return self.func(x1,x2,x3,x4)

    def grad(self, x):
        return approx_fprime(x,self.function,epsilon = 1e-5)

    def gs_fixed_rate(self):
        x = self.x0
        f = self.function(x)
        self.dataset = self.dataset[0:0] # Удалить результаты других методов
        try:
            for i in range(1, self.max_iterations):
                if self.PIR:
                    write({'Iteration': i-1, 'X': x[:self.n_variables], 'f(X)': f})
                if self.SIR:
                    self.dataset.append({'X': x[:self.n_variables], 'f(X)': f}, ignore_index=True)
                diff = -self.lr*self.grad(x)

                if np.all(np.abs(diff) <= self.eps):
                    break
                x += diff
                f = self.function(x)
            if i == self.max_iterations:
                return {'X': x[:self.n_variables], 'f(X)': f, 'Отчет о работе алгоритма': 1}
            else:
                return {'X': x[:self.n_variables], 'f(X)': f, 'Отчет о работе алгоритма': 0}
        except Exception as e:
            return {'X': x[:self.n_variables], 'f(X)': f, 'Отчет о работе алгоритма': 2}


    def gs_splitted_rate(self):
        lr = self.lr
        x = self.x0
        self.dataset = self.dataset[0:0]
        try:
            for i in range(1, self.max_iterations):
                f0 = self.function(x)
                
                if self.PIR:
                    write({'Iteration': i-1, 'x': x[:self.n_variables], 'f(X)': f0})
                if self.SIR:
                    self.dataset.append({'X': x[:self.n_variables], 'f(X)': f}, ignore_index=True)
                diff = -lr*self.grad(x)

                if np.all(np.abs(diff) <= self.eps):
                    break
                x += diff
                f = self.function(x)
                if f0 > f:
                    lr *= 1.5
                else:
                    lr /= 1.5
            if i == self.max_iterations:
                return {'X': x[:self.n_variables], 'f(X)': f, 'Отчет о работе алгоритма': 1}
            else:
                return {'X': x[:self.n_variables], 'f(X)': f, 'Отчет о работе алгоритма': 0}
        except Exception as e:
            return {'X': x[:self.n_variables], 'f(X)': f, 'Отчет о работе алгоритма': 2}

    def gs_optimal_rate(self):
        self.dataset = self.dataset[0:0]
        try:
            x = self.x0
            for i in range(1, self.max_iterations):
                f0 = self.function(x)
                
                if self.PIR:
                    write({'Iteration': i-1, 'x': x[:self.n_variables], 'f(X)': f0})
                if self.SIR:
                    self.dataset.append({'X': x[:self.n_variables], 'f(X)': f}, ignore_index=True)
                lr = minimize_scalar(fun = lambda lr: self.function(x - lr*self.grad(x)), method='brent').x
                diff = -lr*self.grad(x)

                if np.all(np.abs(diff) <= self.eps):
                    break
                x += diff
                f = self.function(x)
            if i == self.max_iterations:
                return {'X': x[:self.n_variables], 'f(X)': f, 'Отчет о работе алгоритма': 1}
            else:
                return {'X': x[:self.n_variables], 'f(X)': f, 'Отчет о работе алгоритма': 0}
        except Exception as e:
            return {'X': x[:self.n_variables], 'f(X)': f, 'Отчет о работе алгоритма': 2}

    def newton_cg(self):
        """
        Препод сказал, больше не нужно реализовывать функции...
        """
        res = minimize(fun = self.function, x0 = self.x0, method='Newton-CG', jac = self.grad,options= {'maxiter': self.max_iterations, 'xtol': self.eps})
        return {'X': res.x[:self.n_variables], 'f(X)': res.fun}

    def minimize2(self, method = 'Метод сопряженных градиентов'):
        func_dict = {
            'Градиентный спуск с постоянным шагом': self.gs_fixed_rate,
            'Градиентный спуск с дроблением шага': self.gs_splitted_rate,
            'Метод наискорейшего спуска': self.gs_optimal_rate,
            'Метод сопряженных градиентов': self.newton_cg
        }
        return func_dict[method]()
    
    

if __name__ == '__main__':
    print(GradientMethod(func = 'x1**2 + (x2+5)**2', n_variables = 2, x0 = '[10,10]', max_iterations = 100, eps = 1e-5, SIR = True, PIR = True, lr = 0.1).minimize())

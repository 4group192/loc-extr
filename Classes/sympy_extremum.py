from sympy import *
import numpy as np
import timeit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px

class Extremum:
    def __init__(self, variables: str, func, limits=None):
        if limits is None:
            self.x = np.arange(-1000, 1000, 0.1)
            self.y = np.arange(-1000, 1000, 0.1)
        else:
            if limits[0] is not None:
                self.x = np.arange(limits[0][0], limits[0][1], 0.1)
            if limits[1] is not None:
                self.y = np.arange(limits[1][0], limits[1][1], 0.1)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.z = func(self.X, self.Y)
        self.symb_1, self.symb_2 = symbols(variables)
        self.analytic_func = func(self.symb_1, self.symb_2)

    def extremums(self):
        diff_x = self.analytic_func.diff(self.symb_1)
        diff_y = self.analytic_func.diff(self.symb_2)
        critical_values = solve([diff_x, diff_y], [self.symb_1, self.symb_2])

        diff_xx = diff_x.diff(self.symb_1)
        diff_xy = diff_x.diff(self.symb_2)
        diff_yy = diff_y.diff(self.symb_2)

        ans = {'Max': [], 'Min': [], 'Требуется дополнительнок исследование': []}
        func_silv = diff_xx * diff_yy - diff_xy**2
        for point in critical_values:
            A = diff_xx.subs(self.symb_1, point[0]).subs(self.symb_2, point[1])
            val = func_silv.subs(self.symb_1, point[0]).subs(self.symb_2, point[1])
            point = {'x': point[0], 'y': point[1], 'z': val}
            if val < 0:
                ans['Требуется дополнительнок исследование'].append(point)
            else:
                if A > 0:
                    ans['Min'].append(point)
                else:
                    ans['Max'].append(point)
        return ans

    def visualize(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(self.X, self.Y, self.z)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    
    def visualize2(self):
        ans = self.extremums().items()
        x, y, z = [], [], []
        for kind in ans:
            for point in kind[1]:
                x.append(float(point['x']))
                y.append(float(point['y']))
                z.append(float(point['z']))
        
        fig = go.Figure(data=[go.Surface(z=self.z, x=self.x, y=self.y)])
        fig.add_trace(go.Scatter3d(z = z, x = x, y= y))
        fig.update_layout(title=str(self.analytic_func), autosize=False,
                  width=1000, height=1000,
                  margin=dict(l=65, r=50, b=65, t=90))
        return fig


if __name__ == '__main__': 
    Example1 = Extremum('x y', lambda x, y: y*(x**2)+x*(y**3) - x*y, limits=[[-10, 10], [-10, 10]])
    Example1.visualize2().show()
from jinja2 import ModuleLoader
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures

class LinearModels:
    def __init__(self, model = 'classic'):
        self.__model = model

    def get_weights(self):
        return self.__weights

    def fit(self, X, Y,reg = None, alpha = 0.1):
        assert len(X) == len(Y)

        model = self.__model
        self.X = np.array(X) 
        self.Y = np.array(Y)
        X = np.c_[np.ones(len(X)), np.array(X)]
        self.n_features = X.shape[1] - 1
        Y = np.array(Y)

        if model == 'poly':
            X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
        elif model == 'expo':
            Y = np.log(Y)

        if reg is None:
            loss = lambda weights: np.sum((X.dot(weights) - Y)**2)
            self.__weights = minimize(loss, x0 = X.shape[1]*[0]).x
        elif reg == 'l2':
            loss = lambda weights: np.sum((X.dot(weights) - Y)**2) + alpha*np.sum((weights)**2)
            self.__weights = minimize(loss, x0 = X.shape[1]*[0]).x
        elif reg == 'l1':
            loss = lambda weights: np.sum((X.dot(weights) - Y)**2) + alpha*np.sum(np.abs(weights))
            self.__weights = minimize(loss, x0 = X.shape[1]*[0]).x
        if model == 'expo':
            self.__weights = np.exp(self.__weights)
        return self
    def predict(self, X):

        if self.__model == 'poly':
            X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(np.c_[np.ones(len(X)), np.array(X)])

        elif self.__model == 'expo':
            X = np.c_[np.array(X)]
            return self.__weights[0]*self.__weights[1:]**X
        
        else:
            X = np.c_[np.ones(len(X)), np.array(X)]
            
        return X.dot(self.__weights)

    def visualize(self):
        assert self.get_weights is not None, 'Следует сначала обучить модель'
        assert self.n_features <=2, 'Виузализация недоступна: кол-во регрессоров > 2'

        fig = go.Figure()
        if self.n_features == 1:
            x = np.linspace(np.min(self.X),np.max(self.X))
            y = self.predict(x)
            fig.add_trace(go.Scatter(x = x.reshape(1,-1)[0], y = y.reshape(1,-1)[0], name = 'y_p'))
            fig.add_trace(go.Scatter(x = self.X, y = self.Y, mode = 'markers', name = 'y'))
        else:
            x1, x2 = np.linspace(np.min(self.X),np.max(self.X)), np.linspace(np.min(self.X),np.max(self.X))
            X1, X2 =np.meshgrid(x1, x2)
            y = self.__weights[0] + X1*self.__weights[1] + self.__weights[2]
            fig = go.Figure(data=[go.Surface(
                x = x1,
                y = x2,
                z = y
            )])
            fig.add_trace(go.Scatter3d(
                x = self.X[:,0],
                y = self.X[:,1],
                z = self.Y
            ))
        return fig


            

if __name__ == '__main__':
    

    poly = LinearModels('classic')
    poly.fit(X = [1,2,3,4],Y = [5,6,7,8])
    print(poly.get_weights())
    poly.predict([1,2,3,4])
    poly.visualize().show()
    

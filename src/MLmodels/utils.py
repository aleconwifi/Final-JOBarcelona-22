import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

class Utils(object):

    def func_overX(self, X):
        Y = []
        for element in X:
            Y.append(sum(element.flatten()) > element.flatten().shape[0] // 2)
        return np.asarray(Y)

    def _ensure_dimensionalit(self, arr):
        return arr if len(arr[0].shape) == 1 else [x.flatten() for x in arr]

    def _acc(self, y_pred, y_target):

        if type(y_pred) == np.array and type(y_target) == np.array:
            assert(y_pred.shape == y_target.shape)
            mask = y_pred == y_target

        else:
            assert(len(y_pred) == len(y_target))
            mask = [x == y for x, y in zip(y_pred, y_target)]

        return sum(mask)/len(mask)

    def mse(self, y_pred, y_target):
        return mean_squared_error(y_target, y_pred)

    def do_scaling(self, X):
        Scaler = StandardScaler()
        data_scaled = Scaler.fit_transform(X)
        return Scaler, data_scaled

    def _gen_gridSearch(self, model, hyperparams, n_splits=3):

        cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
        grid = GridSearchCV(model, param_grid=hyperparams, scoring = 'neg_mean_absolute_error',
                            cv = cv, n_jobs= -1, verbose=2, return_train_score=True) 

        return grid

    def df_Grid(self):
        if self.grid_flag:
            c = self.grid.__dict__['cv_results_']['params']
            a = ['params'] + \
                [f'split{n}_test_score' for n in range(self.n_splits)]
            data = pd.DataFrame({h: i for h, i in zip(
                a, (c, *[self.grid.__dict__['cv_results_'][f'split{n}_test_score'] for n in range(self.n_splits)]))})
            
            data[list(data['params'][0].keys())] = pd.DataFrame(data['params'].tolist())
            
            return data
        else:
            print('Grid has not been calculated')
                       
    def ci(self, alpha):
        def f(x):
            return scipy.stats.t.interval(alpha = alpha, df = self.n_splits - 1, loc = x['mean'], scale = x['sem'])
        return f

    def top_params(self, alpha = 0.95, n = None): #retorna els parametres amb millor ci acc
        df = self.df_Grid()
        df['mean'] = df.filter(regex='test').mean(axis = 1) #agafem columnes nombrades 'split*' calculem mitja
        df['sem'] = df.filter(regex='test').apply(scipy.stats.sem, axis = 1) + 1e-8 #standard error of mean
        #df['ci'] = df.apply(self.ci(alpha), axis = 1)
        df['sort'] = [0.5 * x[1] - abs(x[0] - x[1]) * 0.5 for x in df['ci']] 
        df = df.sort_values('sort', ascending=False)
        
        if n:
            return df[:n]
        return df[:]

    def boxplots(self, n_params=10, duplicates=False):  # for top_params_df
        df = self.top_params(0.95, n_params)
        
        df['name'] = df[list(pd.DataFrame(df['params'].tolist()))].astype(str).agg('-'.join, axis=1)
        df = df.loc[:, df.columns.str.contains('score|name')].set_index('name')
        if not duplicates:
            df.drop_duplicates(inplace = True)
        df = pd.DataFrame(pd.DataFrame(df.unstack('name'), columns=[
                        'value']).droplevel(0)).reset_index(level=0)

        sns.set(font='Gill Sans', font_scale=1.2,
                palette='pastel', style="whitegrid")

        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)

        # Plot with horizontal boxes
        sns.boxplot(x='value', y='name', data=df, width=0.6)

        # Tweak the visual presentation
        ax.xaxis.grid(True)
        ax.set(xlabel="Accuracy", ylabel="")
        sns.despine(trim=True, left=True, bottom=True)
        plt.title(f"{n_params} Splits Boxplots {str(self.model.__class__).split('.')[-1][:-2]}")
        plt.show()
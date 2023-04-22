from scipy.optimize import curve_fit
from scipy.special import erf
import numpy as np
import streamlit as st
import math
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Any, Optional
from PIL import Image


class Functions():
    '''
    Function to approximation
    '''
    @staticmethod
    def linear(x: float,a: float, b: float)-> float:
        '''
        y(x) = ax+b
        '''
        return a*x+b
    @staticmethod
    def quadratic(x: float,a: float, b: float, c: float)-> float:
        '''
        y(x) = ax^2+bx+c
        '''
        return a*x*x+b*x +c
    @staticmethod
    def cubic(x:float,a: float, b:float, c: float, d:float)-> float:
        '''
        y(x) = ax^3+bx^2+cx+d
        '''
        return a*x*x*x+b*x*x +c*x + d
    @staticmethod
    def reciprocal(x: float, a: float, b: float)-> float:
        '''
        y(x) = \\frac{a}{x} + b
        '''
        return a/x+b
    @staticmethod
    def exponential(x:float, a: float, b: float, c: float)-> float:
        '''
        y = ae^{x}+b
        '''
        return a*np.exp(b*x)+c
    @staticmethod
    def sinusoid(x: float, a: float, b: float, c: float)-> float:
        '''
        y(x) = a\\sin(bx)+c
        '''
        return a*np.sin(b*x)+c
    @staticmethod
    def gauss_error(x: float, a: float, b: float, c: float)-> float:
        '''
        y(x) = a*erf(bx)+c
        '''
        return a*erf(b*x)+c

@dataclass
class Approximations:
    '''
    Data and methods require to do approximation
    '''
    names_class_function: list[staticmethod] = None
    names_function: tuple[str] = None
    parameters_name: list[str] = None
    df_from_csv : pd.DataFrame = None
    xdata: pd.DataFrame = None
    ydata: pd.DataFrame = None
    ypred: list[float] = None
    parameters: list[float] = None
    pstats: list[float] = None

    def do_approximation(self, function_options, option: str):
        '''
        approximation using Levenberg-Marquardt algorithm
        '''
        self.parameters, self.pstats = curve_fit(function_options[option] , self.xdata, self.ydata)
  
    def calculate_approximated(self, function_options,coef_name, option: str):
        '''
        approximated values
        '''
        self.ypred = function_options[option](self.xdata, *coef_name.values())


class AppUI:
    
    def __init__(self):
        self.coef_name: dict = {}
        self.option: Optional[str] = None
        self.function_options: Optional[dict] = None
        self.scope: float = 10
        self.factor: float = 1.0
        self.buttons = None
    def set_introduction(self):
        '''
        introduction description
        '''
        st.title("Function Approximation")
        st.markdown('''Upload data from a CSV file where the first column
                    represents independent X variables and the second column
                    represents dependent Y variables.''')
        
        uploaded_file = st.file_uploader("Choose a file", type='csv')
        return uploaded_file
    
    def set_main_part(self, fig, df_1):
        '''
        set plots, tabels etc.
        '''
        st.plotly_chart(fig)
        st.table(df_1)

    def select_option(self):
        '''
        selected options, selected staticmethods, functions
        '''
        with st.sidebar:
            st.header('''Approximation using Levenberg-Marquardt Algorithm''')
            image = Image.open('logoAP.png')
            st.image(image)
            image.close()
            self.option = st.selectbox('Select function',self.function_options.keys())
            st.latex(self.function_options[self.option].__doc__)
            
        #popt, pcov = curve_fit(self.function_options[option] , xdata, ydata)
        #alphabet = ['a', 'b', 'c', 'd', 'e', 'f']
    def set_parameters(self, parameters, alphabet):
        '''
        parameters mapping
        '''
        for i,j in zip(parameters, alphabet):
            coef = float(i)
            with st.sidebar:
                self.coef_name[j] = st.slider(j, (coef - coef*self.scope)*self.factor, (coef + coef*self.scope)*self.factor, coef)

    def add_initial(self)-> None:
        '''
        add initials to end of sidebar
        '''
        with st.sidebar:
            st.divider()
            st.caption('App created by SYMKO')
            
            
class MakeVizualization:
    
    def __init__(self):
        self.template: Optional[str] =  'presentation'
        
    def scatter_and_line_plot(self, xdata: list[float], ydata: list[float], ypred: list[float]):
        '''
        simple plot using plotly
        '''
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=xdata, y=ydata,
                            mode='lines+markers',
                            name='real_data'))
        fig.add_trace(go.Scatter(x=xdata, y=ypred,
                            mode='lines+markers',
                            name='approximation'))
        fig.update_layout(
            title="Approximation plot",
            xaxis_title="X data",
            yaxis_title="Y data")
        fig.update_layout(template=self.template)
        return fig
    
    def tabel(self, col_1, col_2, col_3):
        '''
        dataframe from pandas
        '''
        df = pd.DataFrame(
            {
                "x_data": col_1,
                "y_data": col_2,
                "y_approx": col_3
            }
        )
        return df

def main():
    
    appui = AppUI()
    approx = Approximations()
    vizualize = MakeVizualization()
    
    #approx.parameters_name = ['p1', 'p2', 'p3','p4','p5','p6']
    approx.parameters_name = ['a', 'b', 'c','d','e','f']
    
    uploaded_file = appui.set_introduction()
        
    try:
        df_from_csv= pd.read_csv(uploaded_file)
        first_column = df_from_csv.columns.tolist()[0]
        df_from_csv = df_from_csv.sort_values(by=first_column,ignore_index= True)

        approx.xdata = df_from_csv.iloc[:,0]
        approx.ydata = df_from_csv.iloc[:,1]
        
        approx.names_class_function = [getattr(Functions, m) for m in dir(Functions) if not m.startswith('__')]
        approx.names_function = [a.__name__ for a in approx.names_class_function]
        
        appui.function_options = dict(zip(approx.names_function, approx.names_class_function))
        appui.select_option()
        
        approx.do_approximation(appui.function_options, appui.option)
        appui.set_parameters(approx.parameters, approx.parameters_name)
        approx.calculate_approximated(appui.function_options,appui.coef_name, appui.option)
        
        fig = vizualize.scatter_and_line_plot(approx.xdata, approx.ydata, approx.ypred)
        df_1 = vizualize.tabel(approx.xdata, approx.ydata, approx.ypred)
        
        appui.set_main_part(fig, df_1)
        appui.add_initial()
    except Exception as ValueError:
        st.info('No csv loaded. Please select file')
    
if __name__ == '__main__':
    main()

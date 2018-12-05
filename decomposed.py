import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.style as st
import matplotlib
matplotlib.use('Agg')
import io
import base64
from pylab import rcParams

st.use('fivethirtyeight')

# rcParams['figure.figsize'] = 13, 7
rcParams['figure.figsize'] = 18, 8
rcParams['xtick.major.pad']='4'
rcParams['ytick.major.pad']='4'
rcParams['lines.linewidth'] = 1
rcParams['savefig.facecolor'] = "1"
rcParams['axes.facecolor']= "1"
rcParams["axes.edgecolor"] = "1"

from stldecompose import decompose, forecast

class Decomposed(object):

    def dviz(data, yl):
        
        plt.figure(facecolor='#ffffff')
        plt.gcf().clear()
        # data.plot('o-', marker='o', color='b')
        plt.plot(data, 'o-', marker='.', color='b')
        plt.xticks(rotation=30) 
        plt.grid(True, color='#e5e5cc', linestyle='-', linewidth=1)
        plt.xlabel(None)
        plt.ylabel(str(yl) + ' Sales Averages (USD)', labelpad=20)
        plt.title('Dataset Visualization', y=1.05, color='#630b0b', fontsize=21)
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format = 'png')
        plot = base64.b64encode(img.getvalue()).decode()
        return plot

    def trend(data, period, yl):
        stl = decompose(data, period=len(data)) 
        plt.figure(facecolor='#ffffff')
        plt.gcf().clear()
        plt.plot(stl.trend, 'o-', marker='o', color='b')
        plt.xticks(rotation=30) 
        plt.grid(True, color='#e5e5cc', linestyle='-', linewidth=1)
        plt.xlabel(None)
        plt.ylabel(str(yl) + ' Sales Averages (USD)', labelpad=20)
        plt.title('Sales Trends', y=1.05, color='#630b0b', fontsize=21)
        plt.tight_layout()

#         img = io.BytesIO()
#         plt.savefig(img, format = 'png')
#         plot = base64.b64encode(img.getvalue()).decode()
#         return plot
        return plt.show
    
    def season(data, period, yl):
        # v = int(len(data)//(len(data)//4))
        stl = decompose(data, period=period)
        plt.gcf().clear()
        plt.plot(stl.seasonal, '-', color='b')

        # for i,j in zip(stl.seasonal.index, stl.seasonal.values):
        #     plt.annotate(str(j),xy=(i,j))

        plt.xticks(rotation=30) 
        plt.grid(True)
        plt.xlabel(None)
        plt.ylabel(str(yl) + ' Sales Averages (USD)', labelpad=20)
        plt.title('Sales Seasonality', y=1.05, color='#630b0b', fontsize=21)
        plt.tight_layout()
        
#         img = io.BytesIO()
#         plt.savefig(img, format = 'png')
#         plot = base64.b64encode(img.getvalue()).decode()
        return plt.show


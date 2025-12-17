# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 12:00:37 2025

@author: cting
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def plot_prices(stock, permno, figfilepath="test.png"):
    xdata = np.array(pd.to_datetime(stock.date, format='%Y%m%d'))
    xlims = mdates.date2num([xdata[0], xdata[-1]])
    ydata = stock.price

    # Construct the mesh for setting gradient
    xv, zv = np.meshgrid(np.linspace(0,5000,100), np.linspace(0,5000,100))
    
    nmax = len(str(int(ydata.max())))
    factor = 10**(nmax-1)
    ymax = np.ceil (np.ceil (ydata.max())/factor)*factor

    # Draw the image over the whole plot area
    fig, ax = plt.subplots(figsize=(12,8))
    ax.imshow(zv, cmap='Oranges', origin='lower', 
              extent=[xlims[0], xlims[1], 0, ymax], aspect='auto', alpha=0.7)
    # Set the range of y axis
    ax.set_ylim(0, 1.05*ydata.max())
    
    # Erase above the data by filling with white
    ax.fill_between(xdata, ydata, ymax, color='white')
    
    # Line plot 
    ax.plot(xdata, ydata, 'r-', linewidth=1)
    
    plt.ylabel('$')
    plt.xlabel('')
    plt.grid()
    
    legstr = str(permno) + ' ' +  str(stock.TICKER.iloc[-1])
    plt.legend([legstr], loc="upper left")
    fig.savefig(figfilepath, dpi=300)
    plt.show()

###############################################################################

datadir = 'C:\\Users\\scarl\\Documents\\Research\\data\\'

permno = 80837

filename = datadir + str(permno) + '.csv'

stock = pd.read_csv(filename, usecols=['date', 'COMNAM', 'PERMCO',
            'TICKER', 'PRIMEXCH', 'SHRCLS', 'SHROUT',
            'OPENPRC', 'BIDLO', 'ASKHI', 'PRC', 'VOL',
            'CFACPR', 'CFACSHR', 'RETX','RET', 
            'BID', 'ASK', 'vwretx'],
            converters={'RETX':str, 'RET':str, 'SHRCLS':str}) 

price = stock.PRC/stock.CFACPR
stock['price'] = abs(price)
n = len(price)
print(n)
for i in range(n):
    if stock.RETX[i].isalpha() == True or stock.iloc[i].RET == '':
        print(i, stock.date.iloc[i])

outputfile = str(permno) + '.png'    
plot_prices(stock, permno, figfilepath=outputfile)
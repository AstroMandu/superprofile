
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

from class_hermodel import GHh3model
from herlogp import model
import pandas as pd
import pylab as plt 

from herroutines import bound_to_lowspn, bound_to_loglowlogspn, linmap, logmap, invlinmap, invlogmap
    



def main(xx,yy,e_y):
    xx,yy,e_y = df['x'],df['y'],df['e_y']
    maxy = np.max(yy)
    stdy = np.std(yy)
    minx,maxx = np.min(xx),np.max(xx)

    hermodel = GHh3model(xx,yy,1./e_y)
    hermodel.dict_bounds = {
        'A':bound_to_lowspn([0,maxy]),
        'B':bound_to_lowspn([minx,maxx]),
        'C':bound_to_loglowlogspn([1,100]),
        'Z':bound_to_lowspn([-stdy,stdy])
    }

    guess = [linmap(maxy, hermodel.dict_bounds['A']),
            linmap(xx[np.argmax(yy)], hermodel.dict_bounds['B']),
            logmap(10, hermodel.dict_bounds['C']),
            0,
            linmap(0, hermodel.dict_bounds['Z'])]

    logP = hermodel.make_logP()

    res = minimize(logP, guess, method='Nelder-Mead')
    
    A  = invlinmap(res.x[0], hermodel.dict_bounds['A'])
    B  = invlinmap(res.x[1], hermodel.dict_bounds['B'])
    C  = invlogmap(res.x[2], hermodel.dict_bounds['C'])
    h3 = res.x[3]
    Z  = invlinmap(res.x[4], hermodel.dict_bounds['Z'])
    
    return (A,B,C,h3,Z)


if __name__=='__main__':    
    df = pd.read_csv('/home/mskim/workspace/research/data/Rory/varywind3/RPfiles_0.4mJybeam/snap22_RP45_LOS45/df_stacked.csv', sep='\\s+')
    
    xx,yy,e_y = df['x'],df['y'],df['e_y']
    fig, ax = plt.subplots()
    ax.step(df['x'],df['y'], where='mid')
    
    # A,B,C,h3,Z = main(df['x'],df['y'],df['e_y'])

    # xxx = np.ascontiguousarray(np.linspace(df['x'].min(),df['x'].max(),1000))
    # yyy = model(xxx,A,B,1./C,h3,Z)
    # ax.plot(xxx,yyy)
    
    # ax.axvline(B)
    # ax.axvline(xxx[np.argmax(yyy)])

    cs = CubicSpline(xx, yy)
    xxx_cs = np.linspace(xx.min(), xx.max(), 1000)
    ax.plot(xxx_cs, cs(xxx_cs), label='cubic spline')
    ax.legend()

    plt.show()
    
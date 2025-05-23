import pylab as plt
import numpy as np


class Plotter:
    
    def __init__(self, xx_orig, xx_shft, yy_orig, yy_shft):
        
        self.xx_orig = xx_orig
        self.xx_shft = xx_shft
        self.yy_orig = yy_orig
        self.yy_shft = yy_shft
        
        self.dict_comp = {}
        return
    
    def gauss(self, ampl, velo, disp):
        
        return ampl * np.exp(-np.power(self.xx_orig-velo, 2.) / (2 * np.power(disp, 2.)))
    
    def add_Gcomp(self, ampl, velo, disp, base):
        
        index = len(self.dict_comp)
        self.dict_comp[index] = self.gauss(ampl, velo, disp)
    
    def plot(self):
        
        fig, axs = plt.subplots(nrows=2)
        axs[0].step(self.xx_orig,self.yy)
        
        for k,v in self.dict_comp.items():
            axs[0].plot(self.xx_orig, v)
            
        axs[1].step(self.xx_shft, self.yy_shft)
            
        plt.show()
            


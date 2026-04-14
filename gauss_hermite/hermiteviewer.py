title = 'hermiteviewer Mk.Ia'

# 22.09.16.
# Minsu Kim @ Sejong Univ
# mandu447@gmail.com

# DEPENDENCIES
# sudo apt install python-tk
# pip3 install numpy
# pip3 install matplotlib
# pip3 install astropy
# pip3 install spectral_cube

import glob
import os
import sys
from tkinter import *
from tkinter import filedialog, messagebox, ttk

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from spectral_cube import SpectralCube
import astropy.units as u

dict_params = {'cursor_xy':(-1,-1), 'multiplier_cube':1000.0, 'unit_cube':r'mJy$\,$beam$^{-1}$', 'multiplier_spectral_axis':0.001}
dict_data = {}
dict_obj  = {}
dict_plot = {'fix_cursor':False}
plt.rcParams["hatch.linewidth"] = 4
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple']

def hermite(params):
    a, b, c, h3, Z = params
    y = (dict_data['spectral_axis']-b)/c
    return a*np.exp(-0.5*y**2)*(1 + h3/np.sqrt(6)*(2*np.sqrt(2)*y**3-3*np.sqrt(2)*y)) + Z

def colorbar(img, spacing=0, cbarwidth=0.01, orientation='vertical', pos='right', label='', ticks=[0], fontsize=13):

    ax = img.axes
    fig = ax.figure
    if(orientation=='vertical'):
        if(pos=='right'):
            cax = fig.add_axes([ax.get_position().x1+spacing, ax.get_position().y0, cbarwidth, ax.get_position().height])
        elif(pos=='left'):
            cax = fig.add_axes([ax.get_position().x0-spacing-cbarwidth, ax.get_position().y0, cbarwidth, ax.get_position().height])
            cax.yaxis.set_ticks_position('left')
    elif(orientation=='horizontal'):
        if(pos=='top'):
            cax = fig.add_axes([ax.get_position().x0, ax.get_position().y1+spacing, ax.get_position().width, cbarwidth])
            cax.tick_params(axis='x', labelbottom=False, labeltop=True)

            # cax.xaxis.tick_top()
        elif(pos=='bottom'):
            cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0-spacing-cbarwidth, ax.get_position().width, cbarwidth])
    
    if(len(ticks)!=1):
        cbar = plt.colorbar(img, cax=cax, orientation=orientation, ticks=ticks)
    else: cbar = plt.colorbar(img, cax=cax, orientation=orientation)
    cbar.set_label(label=label, fontsize=fontsize)
    return cbar, cax

def label_panel(ax, text, xpos=0.05, ypos=0.95, color='black', fontsize=10, inside_box=False, pad=5.0):
    # MAKES A LABEL ON GIVEN PANEL

    # PARAMETERS:
    # ax: matplotlib ax
    # text: (str) message to write
    # xpos: xpos of labelbox (relative corrdinates, 0 to 1)
    # ypos: ypos of labelbox (relative corrdinates, 0 to 1)
    # color: color of the text
    # fontsize=: fontsize of the text
    # inside_box = whether to write the text inside a box
    # pad: space between the text and the surrounding box

    # RETURNS:
    # Nothing

    if(inside_box==True):
        ax.text(xpos, ypos, text, transform=ax.transAxes,
            fontsize=fontsize, color=color, verticalalignment='top', 
            bbox=dict(facecolor='none', edgecolor=color, pad=pad))
    else:
        ax.text(xpos, ypos, text, transform=ax.transAxes,
            fontsize=fontsize, color=color, verticalalignment='top', 
            bbox=dict(facecolor='none', edgecolor='none', pad=pad))

def fillentry(entry, content):
    if(entry['state']=='readonly'):
        entry['state']='normal'
        entry.delete(0, "end")
        entry.insert(0, content)
        entry['state']='readonly'
    else:
        entry.delete(0, "end")
        entry.insert(0, content)

def makelabelentry(frame, array, title, startcol, widthlabel, widthentry):
    if(len(title)==0):
        title=array
    for i, content in enumerate(array):
        globals()['label_%s'%(content)] = Label(frame, text=title[i], width=widthlabel, anchor='e')
        globals()['label_%s'%(content)].grid(row=i+startcol, column=0, padx=5)
        globals()['entry_%s'%(content)] = Entry(frame, width=widthentry, justify='right')
        globals()['entry_%s'%(content)].grid(row=i+startcol, column=1)


def initdisplay():

    def _clear(canvas):
        for item in canvas.get_tk_widget().find_all():
            canvas.get_tk_widget().delete(item)

    if 'fig1' not in dict_plot:

        fig1, ax1 = plt.subplots()#tight_layout=True)
        fig1.set_figwidth(500/fig1.dpi)
        fig1.set_figheight(460/fig1.dpi)
        fig1.subplots_adjust(left=0.1, right=0.90, top=0.99, bottom=0.05)

        canvas1 = FigureCanvasTkAgg(fig1, master=dict_obj['frame_display'])   #DRAWING FIGURES ON GUI FRAME
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=TOP)#, fill=BOTH, expand=True)
        fig1.canvas.mpl_connect('motion_notify_event', tracecursor)  #CONNECTING MOUSE CLICK ACTION
        fig1.canvas.mpl_connect('scroll_event', zoom)

        fig2, (ax2, ax3) = plt.subplots(nrows=2, sharex=True)
        fig2.set_figwidth(500/fig2.dpi)
        fig2.set_figheight(500/fig2.dpi)
        fig2.subplots_adjust(hspace=0, top=0.96, bottom=0.16)

        ax2.plot(dict_data['spectral_axis'], np.zeros_like(dict_data['spectral_axis']))

        canvas2 = FigureCanvasTkAgg(fig2, master=dict_obj['frame_line'])
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

        dict_plot['fig1']    = fig1
        dict_plot['ax1']     = ax1
        dict_plot['canvas1'] = canvas1

        dict_plot['fig2']    = fig2
        dict_plot['ax2']     = ax2
        dict_plot['ax3']     = ax3
        dict_plot['canvas2'] = canvas2

        dict_params['drawnew'] = False

    elif(dict_params['drawnew']):
        del dict_plot['img1']
        del dict_plot['fig1']

        dict_obj['frame_display'].destroy()
        dict_obj['frame_display'] = Frame(frame_L, height=500, width=500, bg='white')
        dict_obj['frame_display'].pack(side='top')

        dict_obj['frame_mapselect'].pack_forget()
        dict_obj['frame_mapselect'].pack(fill=BOTH, expand=True)

        dict_obj['frame_line'].destroy()
        dict_obj['frame_line'] = Frame(frame_R, width=500,height=500, bg='white')
        dict_obj['frame_line'].pack()

        initdisplay()

    if 'img1' in dict_plot:
        cur_xlim = dict_plot['ax1'].get_xlim()
        cur_ylim = dict_plot['ax1'].get_ylim()
        dict_plot['ax1'].clear()
        dict_plot['ax1'].set_xlim(cur_xlim)
        dict_plot['ax1'].set_ylim(cur_ylim)
    else:
        dict_plot['ax1'].clear()
    
    path_map = glob.glob(dict_params['path_hermite'])[0]
    dict_plot['img1'] = dict_plot['ax1'].imshow(np.load(path_map)[dict_plot['index_plot'],:,:], interpolation='none', cmap=dict_plot['cmap'])
    ylim = dict_plot['ax1'].get_ylim()
    if(ylim[0]>ylim[1]):
        dict_plot['ax1'].invert_yaxis()
    if(dict_plot['fix_cursor']):
        cross = dict_plot['ax1'].scatter(dict_params['cursor_xy'][0], dict_params['cursor_xy'][1], marker='+', color='white', s=100)
        cross.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='black')])
    if 'cax' in dict_plot:
        del dict_plot['cax']
    _,dict_plot['cax'] = colorbar(dict_plot['img1'], cbarwidth=0.03)
    dict_plot['canvas1'].draw()

    dict_plot['ax2'].clear()
    dict_plot['ax3'].clear()
    
    # dict_plot['ax2'].plot(dict_data['spectral_axis'], np.zeros_like(dict_data['spectral_axis']))
    # dict_plot['ax3'].plot(dict_data['spectral_axis'], np.zeros_like(dict_data['spectral_axis']))

    dict_plot['canvas2'].draw()

    drawplots()


    # plt.close(fig)


def readdata(path_cube=None, path_hermite=None):

    if(path_cube!=None):
        dict_params['path_cube'] = path_cube
    if(path_hermite!=None):
        dict_params['path_hermite'] = path_hermite

    dict_data['cube'] = fits.getdata(dict_params['path_cube'])*dict_params['multiplier_cube']
    dict_data['spectral_axis'] = (SpectralCube.read(dict_params['path_cube'])).with_spectral_unit(u.m/u.s, velocity_convention='optical').spectral_axis.value*dict_params['multiplier_spectral_axis']

    dict_plot['index_plot'] = 0
    dict_plot['cmap'] = 'Blues'
    dict_data['imsize'] = dict_data['cube'][0,:,:].shape

    # data_her = fits.getdata(dict_params['path_hermite'])
    data_her = np.load(dict_params['path_hermite'])

    dict_data['A'] = data_her[0,:,:] * dict_params['multiplier_cube']
    dict_data['B'] = data_her[1,:,:] *dict_params['multiplier_spectral_axis']
    dict_data['C'] = data_her[2,:,:] *dict_params['multiplier_spectral_axis']
    dict_data['h3']= data_her[3,:,:]
    dict_data['Z'] = data_her[4,:,:] * dict_params['multiplier_cube']

    del data_her

    dict_params['drawnew'] = True
    initdisplay()


def loaddata():

    def browse_cube():
        path_cube = filedialog.askopenfilename(title='Path to cube', filetypes=[('FITS file', '.fits .FITS')])
        if(len(path_cube)==0): return

        if(len(fits.getdata(path_cube).shape)<3 or len(SpectralCube.read(path_cube).spectral_axis)==1):
            messagebox.showerror("Error", "Cube should have at least three dimensions.")
            return
        
        fillentry(entry_path_cube, path_cube)

        possible_path_hermite = glob.glob(os.path.dirname(path_cube)+'/hermite.npy')
        if(len(possible_path_hermite)==1):
            browse_hermite(possible_path_hermite[0])

    def browse_hermite(path_hermite=None, initialdir=None):
        if(path_hermite==None):
            path_hermite = filedialog.askopenfilename(title='Path to hermite', initialdir=initialdir)
            if(len(path_hermite)==0): return

        fillentry(entry_path_hermite, path_hermite)  

    def btncmd_toplv_browse_cube():
        browse_cube()

    def btncmd_toplv_browse_hermite():
        browse_hermite()

    def btncmd_toplv_apply():
        dict_params['path_cube'] = entry_path_cube.get()
        dict_params['path_hermite'] = entry_path_hermite.get()
        readdata()

        dict_plot['toplv'].destroy()
   

    def btncmd_toplv_cancel():
        toplv.destroy()

    toplv = Toplevel(root)

    frame_toplv1 = Frame(toplv)
    frame_toplv2 = Frame(toplv)

    makelabelentry(frame_toplv1, ['path_cube', 'path_hermite'], [], 0, 20, 20)

    btn_toplv_browsecube = Button(frame_toplv1, text='Browse', command=btncmd_toplv_browse_cube)
    btn_toplv_browsecube.grid(row=0, column=2)

    btn_toplv_browsehermite = Button(frame_toplv1, text='Browse', command=btncmd_toplv_browse_hermite)
    btn_toplv_browsehermite.grid(row=1, column=2)

    ttk.Separator(frame_toplv2, orient='horizontal').pack(fill=BOTH)

    btn_toplv_apply = Button(frame_toplv2, text='Apply', command=btncmd_toplv_apply)
    btn_toplv_cancel = Button(frame_toplv2, text='Cancel', command=btncmd_toplv_cancel)
    btn_toplv_cancel.pack(side='right')
    btn_toplv_apply.pack(side='right')

    frame_toplv1.pack()
    frame_toplv2.pack(fill=BOTH)

    dict_plot['toplv'] = toplv


def apply_mapselect(*args):

    var = dict_plot['var_mapselect'].get()
    # ['Integrated flux', 'SGfit velocity', 'SGfit vdisp', 'Ngauss', 'SGfit Peak S/N']

    if(var=='A'):
        dict_plot['index_plot'] = 0
        dict_plot['cmap'] = 'Blues'
    if(var=='B'):
        dict_plot['index_plot'] = 1
        dict_plot['cmap'] = 'jet'
    if(var=='C'):
        dict_plot['index_plot'] = 2
        dict_plot['cmap'] = 'jet'
    if(var=='h3'):
        dict_plot['index_plot'] = 3
        dict_plot['cmap'] = 'seismic'
    if(var=='Z'):
        dict_plot['index_plot'] = 4
        dict_plot['cmap'] = 'seismic'

    # dict_plot['index_plot'] = int(var)

    initdisplay()

def fix_cursor(event):
    dict_plot['fix_cursor'] = (dict_plot['fix_cursor']+1)%2

    initdisplay()

root = Tk()

root.title(title)
# root.bind("<Return>", lambda x: updatedisplay())
root.resizable(False, False)


menubar = Menu(root)

menu_1 = Menu(menubar, tearoff=0)
menu_1.add_command(label="Load data", command=loaddata)

menu_2 = Menu(menubar, tearoff=0)
menu_2.add_command(label='TBU')

menubar.add_cascade(label="Load", menu=menu_1)
menubar.add_cascade(label="Option", menu=menu_2)

dict_obj['frame_master'] = Frame(root)
frame_L = Frame(dict_obj['frame_master'], height=500, width=500, bg='white')
frame_M = Frame(dict_obj['frame_master'], height=500, width=50, bg='white')
frame_R = Frame(dict_obj['frame_master'], height=500, width=500, bg='white')
dict_obj['frame_display'] = Frame(frame_L, height=500, width=500, bg='white')
dict_obj['frame_display'].pack()

dict_obj['frame_mapselect'] = Frame(frame_L)
OptionList = ['A','B','C','h3','Z']
dict_plot['var_mapselect'] = StringVar()
dict_plot['var_mapselect'].set(OptionList[0])

dropdown_mapselect = OptionMenu(dict_obj['frame_mapselect'], dict_plot['var_mapselect'], *OptionList)
# dropdown_mapselect.config
dropdown_mapselect.pack(side='right')
dict_plot['var_mapselect'].trace("w", apply_mapselect)
dict_obj['frame_mapselect'].pack(fill=BOTH, expand=True)

dict_obj['frame_line'] = Frame(frame_R, width=500,height=500, bg='white')
dict_obj['frame_line'].pack()

frame_L.pack(fill=BOTH, expand=True, side='left')
frame_M.pack(fill=BOTH, expand=True, side='left')
frame_R.pack(fill=BOTH, expand=True, side='right')
dict_obj['frame_master'].pack(fill=BOTH, expand=True)

def drawplots():

    x,y=dict_params['cursor_xy']
    dict_plot['ax2'].clear()
    dict_plot['ax3'].clear()
    # dict_plot['ax2'].plot(dict_data['spectral_axis'], dict_data['cube'][:,y,x])

    dict_plot['ax2'].fill_between(dict_data['spectral_axis'], dict_data['cube'][:,y,x], hatch=r'//', color='lightgray', edgecolor='white')
    dict_plot['ax2'].plot(        dict_data['spectral_axis'], dict_data['cube'][:,y,x], color='lightgray')
    
    A = dict_data['A'][y,x]
    B = dict_data['B'][y,x]
    C = dict_data['C'][y,x]
    h3= dict_data['h3'][y,x]
    Z = dict_data['Z'][y,x]

    ploty = hermite((A, B, C, h3, Z))
    # print(ploty)

    dict_plot['ax2'].plot(dict_data['spectral_axis'], ploty, alpha=0.5)
    dict_plot['ax2'].scatter(B,0, label='Vel_her')

    dict_plot['ax3'].fill_between(dict_data['spectral_axis'], dict_data['cube'][:,y,x]-ploty, hatch=r'//', color='lightgray', edgecolor='white')
    dict_plot['ax3'].plot(        dict_data['spectral_axis'], dict_data['cube'][:,y,x]-ploty, color='lightgray')

    label_panel(dict_plot['ax2'], '(x,y)=({},{}) \nA={:.1e} \nB={:.1e} \nC={:.1e} \nh3={:.1e} \nZ={:.1e}'.format(x,y,A,B,C,h3,Z))
    label_panel(dict_plot['ax3'], 'Residuals')

    # dict_plot['ax2'].set_ylabel('Flux density ({})'.format(dict_params['unit_cube']))
    dict_plot['ax2'].text(-0.12, -0, 'Flux density ({})'.format(dict_params['unit_cube']), ha='center', va='center', transform = dict_plot['ax2'].transAxes, rotation=90)
    dict_plot['ax3'].set_xlabel(r'Spectral axis (km$\,$s$^{-1}$)')

    # canvas_line1 = FigureCanvasTkAgg(fig2, master=frame_line1)
    dict_plot['canvas2'].draw()




def tracecursor(event):
    if(dict_plot['fix_cursor']==False):
        # x,y=event.x, event.y
        if event.inaxes:
            # ax = event.inaxes
            cursor_xy = (round(event.xdata),round(event.ydata))
            # print(cursor_x, cursor_y)

            if(dict_params['cursor_xy']==cursor_xy[0] and dict_params['cursor_xy'][1]==cursor_xy[1]):
                return
            else:
                dict_params['cursor_xy']=cursor_xy
                drawplots()   

def zoom(event):
    cur_xlim = dict_plot['ax1'].get_xlim()
    cur_ylim = dict_plot['ax1'].get_ylim()

    xdata = event.xdata # get event x location
    ydata = event.ydata # get event y location

    base_scale = 2.

    if event.button == 'up':
        # deal with zoom in
        scale_factor = 1 / base_scale
    elif event.button == 'down':
        # deal with zoom out
        scale_factor = base_scale
    else:
        # deal with something that should never happen
        scale_factor = 1
        print(event.button)

    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

    relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
    rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

    dict_plot['ax1'].set_xlim(np.max([xdata - new_width * (1-relx),0]), np.min([xdata + new_width * (relx),dict_data['imsize'][1]-1]))
    dict_plot['ax1'].set_ylim(np.max([ydata - new_height * (1-rely),0]), np.min([ydata + new_height * (rely),dict_data['imsize'][0]-1]))
    # ax.figure.canvas.draw()
    dict_plot['canvas1'].draw()



root.config(menu=menubar)
root.bind('f', fix_cursor)
# print(sys.argv)
if(len(sys.argv)>1):
    readdata(path_cube=sys.argv[1], path_hermite=sys.argv[2])

root.mainloop()


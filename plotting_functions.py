import numpy as np
import pandas as pd
import os, sys, glob
import matplotlib.pyplot as plt
import matplotlib
import scprep
import seaborn as sns
from scipy.stats import ttest_1samp, zscore
import analysis_helpers as helper
from scipy.ndimage import zoom
from config import *

def determine_symbol(pval):
    symbols = {'0.1':'~', '0.05':'*', '0.01':'**', '0.001':'***'}
    if pval <= 0.001:
        return symbols['0.001']
    if pval <= 0.01:
        return symbols['0.01']
    if pval <= 0.05:
        return symbols['0.05']
    if pval <= 0.1:
        return symbols['0.1']
    return 'n.s.'

def make_barplot_points(dataframe, yname, xname, exclude_subs=[], ylim=[-0.04,0.06], outfn=None, title='',plus_bot=0.04, plus_top=0.07, n_iter=1000, sample_alternative='greater',pairwise_alternative='greater'):
    ## already filter the dataframe to give it just the data necessary - the version you are running or whatever
    # so only have to select the yname
    if (len(exclude_subs)==1) and (exclude_subs[0]=='simulation'): 
        include = SIMULATED_SUBS
        cmap=colors_sim
    else:
        include = np.arange(5,26)
        include = [i for i in include if i not in exclude_subs]
        include = [helper.format_subid(i) for i in include]
        cmap=colors_main
    print(include)
    
    # print(include)
    order=['IM','WMP','OMP']
    d = dataframe[(dataframe['subject_id'].isin(include))]
    v_im = d[d[xname]=='IM'][yname].values
    v_wm = d[d[xname]=='WMP'][yname].values
    v_om = d[d[xname]=='OMP'][yname].values
    # print(v_im.shape,v_wm.shape,v_om.shape)
    # print(d[d[xname]=='OMP'].subject_id.unique())
        
    # run stats
    _,p1=ttest_1samp(v_im, popmean=0, alternative=sample_alternative)
    _,p2=ttest_1samp(v_wm, popmean=0, alternative=sample_alternative)
    _,p3=ttest_1samp(v_om, popmean=0, alternative=sample_alternative)
    print(f'IM {np.round(p1,4)}, WM {np.round(p2,4)}, OM {np.round(p3,4)}')

    _,p4,_=helper.permutation_test(np.array([v_im,v_wm]), n_iter, alternative=pairwise_alternative)
    _,p5,_=helper.permutation_test(np.array([v_im,v_om]), n_iter, alternative=pairwise_alternative)
    _,p6,_=helper.permutation_test(np.array([v_wm,v_om]), n_iter, alternative=pairwise_alternative)
    print(f'IMvWM: {np.round(p4,4)}, IMvOM: {np.round(p5,4)}, WMvOM: {np.round(p6,4)}')
    
    pstrs = [determine_symbol(p) for p in [p1,p2,p3,p4,p5,p6]]
    
    fig,ax=plt.subplots(figsize=(4,4))
    sns.barplot(data=d, x=xname, y=yname, palette=[cmap[i] for i in order], order=order, errorbar=None, 
                edgecolor="k",
                errcolor="black",
                errwidth=3, alpha=0.85,ax=ax )
    ax.axhline(0, ls='--', c='k') 
    xlocs=[0,1,2]
    points0, points1, points2 = [], [], []
    for b,subj in enumerate(include):
        try:
            point0 = d[(d['subject_id']==subj) & (d[xname]=='IM')][yname].item()

            # print(point0)
            point1 = d[(d['subject_id']==subj) & (d[xname]=='WMP')][yname].item()
            point2 = d[(d['subject_id']==subj) & (d[xname]=='OMP')][yname].item()
        except:
            print(subj)
            continue
        ax.scatter(xlocs, [point0, point1, point2], c=[colors_sim[i] for i in order], edgecolors='k', zorder=10, linewidths=1, s=30)
        points0.append(point0)
        points1.append(point1)
        points2.append(point2)

        ax.plot(xlocs, [point0, point1, point2], color='k',alpha=0.36, linewidth=0.4)
    
    # draw signif over bar
    xloc_mids = [-0.12, 0.9, 1.9]
    yloc_bar = np.max(np.concatenate([points0,points1,points2]))+0.004
    for i in range(3):
        if pstrs[i] != None:
            plt.text(x=xloc_mids[i], y=yloc_bar, s=pstrs[i], size=12)
    
    # draw signif btwn bar
    xrel_mids = [0.4,1,1.4]
    line_xranges = [[0.2,0.45], [0.2,0.8], [0.55,0.8]]
    line_y = [yloc_bar+plus_bot, yloc_bar+plus_top, yloc_bar+plus_bot]
    
    for i in range(3):
        if pstrs[i+3]!=None:
            plt.axhline(y=line_y[i], xmin=line_xranges[i][0], xmax=line_xranges[i][1], color='k', lw=1)
            plt.text(x=xrel_mids[i], y=line_y[i], s=pstrs[i+3], size=12) 
    

    ax.set_ylabel(yname)
    ax.set_xticklabels(['IM','WM','OM'])
    ax.set(ylim=ylim,title=title,xlabel=xname,ylabel='')
    sns.despine()
    if outfn != None: 
        plt.savefig(outfn, transparent=True,bbox_inches = "tight")
        plt.close()
    else:
        return fig,ax

def make_barplot_errorbar(dataframe, yname, xname, exclude_subs=[], ylim=[-0.04,0.06], outfn=None, title='',plus_bot=0.04, plus_top=0.07, n_iter=1000, sample_alternative='greater',pairwise_alternative='greater'):
    ## already filter the dataframe to give it just the data necessary - the version you are running or whatever
    # so only have to select the yname
    if (len(exclude_subs)==1) and (exclude_subs[0]=='simulation'): 
        include = SIMULATED_SUBS
        cmap=colors_sim
    else:
        include = np.arange(5,26)
        include = [i for i in include if i not in exclude_subs]
        include = [helper.format_subid(i) for i in include]
        cmap=colors_main
    print(include)
    order=['IM','WMP','OMP']
    d = dataframe[(dataframe['subject_id'].isin(include))]
    v_im = d[d[xname]=='IM'][yname].values
    v_wm = d[d[xname]=='WMP'][yname].values
    v_om = d[d[xname]=='OMP'][yname].values
    # print(v_im.shape,v_wm.shape,v_om.shape)
    # print(d[d[xname]=='OMP'].subject_id.unique())
    points0 =d[d[xname]=='IM'][yname].values

    # print(point0)
    points1 = d[d[xname]=='WMP'][yname].values
    points2 = d[d[xname]=='OMP'][yname].values
    # run stats
    _,p1=ttest_1samp(v_im, popmean=0, alternative=sample_alternative)
    _,p2=ttest_1samp(v_wm, popmean=0, alternative=sample_alternative)
    _,p3=ttest_1samp(v_om, popmean=0, alternative=sample_alternative)
    print(f'IM {np.round(p1,4)}, WM {np.round(p2,4)}, OM {np.round(p3,4)}')

    _,p4,_=helper.permutation_test(np.array([v_im,v_wm]), n_iter, alternative=pairwise_alternative)
    _,p5,_=helper.permutation_test(np.array([v_im,v_om]), n_iter, alternative=pairwise_alternative)
    _,p6,_=helper.permutation_test(np.array([v_wm,v_om]), n_iter, alternative=pairwise_alternative)
    print(f'IMvWM: {np.round(p4,4)}, IMvOM: {np.round(p5,4)}, WMvOM: {np.round(p6,4)}')
    
    pstrs = [determine_symbol(p) for p in [p1,p2,p3,p4,p5,p6]]
    
    fig,ax=plt.subplots(figsize=(3,4))
    sns.barplot(data=d, x=xname, y=yname, palette=[cmap[i] for i in order], order=order, errorbar='ci', 
                edgecolor="k",
                errcolor="black",
                errwidth=3, alpha=0.85,ax=ax )
    ax.axhline(0, ls='--', c='k') 
    
    # draw signif over bar
    xloc_mids = [-0.12, 0.9, 1.9]
    yloc_bar = np.max(np.concatenate([points0,points1,points2]))+0.004
    for i in range(3):
        if pstrs[i] != None:
            plt.text(x=xloc_mids[i], y=yloc_bar, s=pstrs[i], size=12)
    
    # draw signif btwn bar
    xrel_mids = [0.4,1,1.4]
    line_xranges = [[0.2,0.45], [0.2,0.8], [0.55,0.8]]
    line_y = [yloc_bar+plus_bot, yloc_bar+plus_top, yloc_bar+plus_bot]
    
    for i in range(3):
        if pstrs[i+3]!=None:
            plt.axhline(y=line_y[i], xmin=line_xranges[i][0], xmax=line_xranges[i][1], color='k', lw=1)
            plt.text(x=xrel_mids[i], y=line_y[i], s=pstrs[i+3], size=12) 
    

    ax.set_ylabel(yname)
    ax.set_xticklabels(['IM','WM','OM'])
    ax.set(ylim=ylim,title=title,xlabel=xname,ylabel='')
    sns.despine()
    if outfn != None: 
        plt.savefig(outfn, transparent=True,bbox_inches = "tight")
        plt.close()

def average_groups(X, V):
    N = len(X)
    group_size = N // V
    averaged_values = np.array([np.mean(X[i*group_size:(i+1)*group_size]) for i in range(V)]) 
    return averaged_values

def rgb2range(RGB):
    r, g, b = RGB[0],RGB[1],RGB[2]
    return (r/256, g/256, b/256)

def transform_values_to_2d_colormap(values0, values1, C0, C1, C2, C3, cmap_resolution=200, plot=True):
    im = np.array([[C0, C1], [C2, C3]])
    zoomed = zoom(im,(cmap_resolution/2,cmap_resolution/2,1), order=1)
    if plot:
        plt.subplot(121)
        plt.imshow(im,interpolation='nearest')
        plt.subplot(122)
        plt.imshow(zoomed,interpolation='nearest') 
        plt.savefig('plots/colorbar.pdf',transparent=True)
    
    # now map values
    minn = np.min([np.min(values0), np.min(values1)])
    maxx = np.max([np.max(values0), np.max(values1)])
    
    m, b = np.polyfit([minn, maxx], [0, cmap_resolution], 1)
    transform = lambda x : x * m + b 
    clip2range_min = lambda V : np.where(V >= 0, V, 0)
    clip2range_max = lambda V : np.where(V < cmap_resolution, V, cmap_resolution-1)
    transform_and_clip = lambda values : clip2range_max(clip2range_min(transform(values)))
    v0_scaled = transform_and_clip(values0).astype(int)
    v1_scaled = transform_and_clip(values1).astype(int)
    final_colors = [zoomed[v0,v1] for v0,v1 in zip(v0_scaled, v1_scaled)]
    return np.array(final_colors)
    
def rgb_to_hex(RGB):
    r, g, b = RGB[0],RGB[1],RGB[2]
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)
    
def create_scatterplot_2dim_colors(tphate_embedding, xlabels, zlabels, colors_4, size=10, cmap_resolution=200, plot=False, outname=None,twoDScatter=False):
    C0, C1, C2, C3 = colors_4
    final_colormap = transform_values_to_2d_colormap(xlabels, zlabels, C0, C1, C2, C3, cmap_resolution=cmap_resolution, plot=plot)
    hex_colors = [rgb_to_hex(c) for c in final_colormap]
    rgb_colors = [rgb2range(c) for c in final_colormap]
    T1 = tphate_embedding[:len(xlabels)]
    if plot == False:
        return T1, hex_colors
    if twoDScatter:
        scprep.plot.scatter2d(T1, c=rgb_colors, ticks=False, alpha=1,s=size,figsize=(3,3),filename=outname,legend=False  )
    else:
        scprep.plot.scatter3d(T1, c=rgb_colors, ticks=False, alpha=1,s=size,figsize=(3,3),filename=outname,legend=False )
    if outname != None:
        plt.close()
    return final_colormap,hex_colors

def run_connected_3D_scatterplot(tphate_embedding, labels1, labels2, colors_4, size=10, cmap_resolution=200, outname=None):
    C0, C1, C2, C3 = colors_4
    final_colormap = transform_values_to_2d_colormap(labels1,labels2, C0, C1, C2, C3, cmap_resolution=cmap_resolution, plot=False)
    hex_colors = [rgb_to_hex(c) for c in final_colormap]
    T1 = tphate_embedding[:len(labels1)]
    rgb_colors = [rgb2range(c) for c in final_colormap]
    plot_3d_scatter_with_lines(T1, rgb_colors, size, outname)
    

def plot_3d_scatter_with_lines(data, c, size=10,outname=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x,y,z=data[:,0],data[:,1],data[:,2]
    # Scatter plot
    ax.scatter(x, y, z, c=c, marker='o', s=size,zorder=2,alpha=1)
    
    # Connect points with lines
    ax.plot(x, y, z, c='k',zorder=1,alpha=0.8)
    # Turn off ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Turn off grid
    ax.grid(False)
    
    # Labels
    ax.set_xlabel('T-PHATE 0')
    ax.set_ylabel('T-PHATE 1')
    ax.set_zlabel('T-PHATE 2')
    if outname:
        plt.savefig(outname, transparent=True,bbox_inches = "tight")
    else:
        plt.show()

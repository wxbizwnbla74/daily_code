#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Class   :
# @description:
# @Time    : 2022/6/2 下午9:50
# @Author  : Duan Ran


import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd

# cellines = ['GM12878']
# for celline in cellines:
#     penguins = sns.load_dataset("penguins")
#     # mydata = pd.read_csv('/Volumes/Samsung_X5/mactop/histonepic.csv'.format(celline),sep=',')
#     mydata = pd.read_csv('/Volumes/Samsung_X5/mactop/comban_motif2.csv'.format(celline), sep=',')
#     # single_list =['Mactop','TopDom','Insulation','Directionality']
#     single_list = ['stable_boundary', 'unstable_boundary', 'mactop']
#
#     sns.set(style="ticks",font_scale=1.5)#设置主题，文本大小
#     flights = mydata.pivot(index="histone",columns= "method", values="value")
#     print(flights)
#     flights2 = flights.loc[:,['stable_boundary', 'unstable_boundary', 'mactop']]
#     f, ax = plt.subplots(figsize=(8, 8))
#
#     print(flights2)
#     flights2.sort_values('stable_boundary', inplace=True, ascending=False)
#     print(flights2)
#     heatmap = sns.heatmap(flights2, linewidths=.5,cmap='Reds',square=True,vmax=.085,vmin=.015)
#     heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=320, ha='left')
#     cbar = heatmap.collections[0].colorbar
#     cbar.set_ticks([.01, .02, .03, .04])
#     # cbar.set_ticklabels(['0%', '20%', '75%', '100%'])
#
#     plt.show()
#     scatter_fig = heatmap.get_figure()
#     # scatter_fig.savefig('./heatmap_3_1/{0}.pdf'.format(celline), dpi=600)


cellines = ['GM12878']
for celline in cellines:
    penguins = sns.load_dataset("penguins")
    # mydata = pd.read_csv('/Volumes/Samsung_X5/mactop/histonepic.csv'.format(celline),sep=',')
    mydata = pd.read_csv('/Volumes/Samsung_X5/mactop/FigThree/demohis.csv'.format(celline), sep=',')
    # single_list =['Mactop','TopDom','Insulation','Directionality']
    single_list = ['com', 'uncom', 'all']

    sns.set(style="ticks", font_scale=1.5)#设置主题，文本大小
    flights = mydata.pivot(index="histone", columns= "method", values="value")
    print(flights)
    # flights2 = flights.loc[:,['Mactop','TopDom','Insulation','Directionality']]
    flights2 = flights.loc[:, ['com', 'uncom', 'all']]
    f, ax = plt.subplots(figsize=(8, 8))

    print(flights2)
    flights2.sort_values('uncom', inplace=True, ascending=False)
    print(flights2)
    heatmap = sns.heatmap(flights2, linewidths=.5, cmap='Reds', square=True, vmax=1.5, vmin=0.5)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=320, ha='left')
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([.01, .02, .03, .04])
    # cbar.set_ticklabels(['0%', '20%', '75%', '100%'])

    plt.show()
    scatter_fig = heatmap.get_figure()
    # scatter_fig.savefig('./heatmap_3_1/{0}.pdf'.format(celline), dpi=600)

#
# ValueError: 'tgb' is not a valid value for name; supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r',
#  'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2',
# 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r',
#  'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',
# 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r',
#  'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu',
# 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral',
# 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r'
# , 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r',
# 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
#

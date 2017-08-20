# Based on an IPython example, but modified to be more compact and applicable to this use case.
# 12-2-2014 - ASJ
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


def colorline(ax,x, y, z, cmap=plt.get_cmap('bwr'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Specify colors in z. Optionally specify a colormap, a norm function and a line width
    '''
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax.add_collection(lc)
    
    return lc
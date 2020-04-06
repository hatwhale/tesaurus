'''Some useful functions for working with generated networks.

- Input-output functions for saving and getting generated results.
- Functions for plotting degree distributions.

Examples
--------
Create 100 random graphs by simple_generate function, save results in '/simple' folder and
plot average cumulative {in+out}-degree distribution in log-log scale.

>>> from dirhypernets import *
>>>
>>> simple_networks = [simple_generate(seed=None, N=500, M=10, sigma=5, beta=0.5, T=0.1) for i in range(100)]
>>> save_networks(simple_networks, 'simple')
>>> degrees = get_degrees('all', *simple_networks)
>>> plot_distribution(*log_cumulative_distribution(degrees, 200), fmt='-', plt_type='cdf')
>>> plt.show()

Plot in-degree distribution of a single graph in log-log scale with its log-binning.

>>> degrees = get_degrees(simple_networks[0])
>>> plot_distribution('in', *degree_distribution(degrees))
>>> plot_distribution(*log_binning(degrees, 30), fmt='o-', plt_type='log')
>>> plt.show()

Get networks data from '/simple' folder and plot average out-degree distribution.

>>> saved_simple_networks = get_networks('simple')
>>> degrees = get_degrees('out', *saved_simple_networks)
>>> plot_distribution(*degree_distribution(degrees), plt_func=plt.plot)
>>> plt.show()

'''

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def save_G_coords(G_coords, edgelist_path, coords_path):
    '''
    Saves network's coordinates and the list of network edges.
    
    Parameters
    ----------
    G_coords : (networkx DiGraph, (numpy array, numpy array[, numpy array]))
        network graph and points coordinates
    edgelist_path : file or string
        file or filename to write the list of edges
    coords_path : file or string
        file or filename to write the coordinates
    
    '''
    G, coords = G_coords
    nx.write_edgelist(G, edgelist_path, data=False)
    cols = ('theta', 'radius') if len(coords) == 2 else ('theta', 'radius_p', 'radius_m')
    pd.DataFrame(dict(zip(cols, coords))).to_csv(coords_path, index=None, sep=' ')

def get_G_coords(edgelist_path, coords_path):
    '''
    Get network's coordinates and the list of network edges.
    
    Parameters
    ----------
    edgelist_path : file or string
        file or filename to write the list of edges
    coords_path : file or string
        file or filename to write the coordinates
    
    Returns
    -------
    networkx DiGraph, (numpy array, numpy array[, numpy array])
        network graph and points coordinates

    '''
    df = pd.read_csv(coords_path, delimiter=' ')
    cols = ('theta', 'radius') if len(df.columns) == 2 else ('theta', 'radius_p', 'radius_m')
    coords = tuple(df[col].values for col in cols)
    G = nx.read_edgelist(edgelist_path, create_using=nx.DiGraph(), nodetype=int)
    G.add_nodes_from([i for i in range(len(df))])
    return G, coords

def save_networks(networks, path):
    '''
    Saves networks coordinates and lists of networks edges.
    
    Parameters
    ----------
    networks : list of (networkx DiGraph, (numpy array, numpy array[, numpy array]))
        networks graphs and coordinates
    path : file or string
        file or filename to write
    
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    elif len(os.listdir(path)) > 0:
        raise Exception("The directory is not empty.")
    edgelist_paths = []
    coords_paths = []
    for i, G_coords in enumerate(networks):
        e_path = os.path.join(path, 'edgelist_{}.txt'.format(i))
        c_path = os.path.join(path, 'coords_{}.txt'.format(i))
        edgelist_paths.append(e_path)
        coords_paths.append(c_path)
        save_G_coords(G_coords, e_path, c_path)

def get_networks(path):
    '''
    Get networks coordinates and lists of networks edges.
    
    Parameters
    ----------
    path : file or string
        file or filename to write
    
    Returns
    -------
    list of (networkx DiGraph, (numpy array, numpy array[, numpy array]))
        networks graphs and coordinates

    '''
    networks = []
    edgelist_paths = sorted([os.path.join(path, p) for p in os.listdir(path) if 'edgelist' in p])
    coords_paths = sorted([os.path.join(path, p) for p in os.listdir(path) if 'coords' in p])
    for e_path, c_path in zip(edgelist_paths, coords_paths):
        networks.append(get_G_coords(e_path, c_path))
    return networks

def get_degrees(deg_type, *networks):
    '''
    Get networks degrees.
    
    Parameters
    ----------
    networks : list of (networkx DiGraph, (numpy array, numpy array[, numpy array]))
        networks graphs and coordinates
    deg_type : {'all', 'in', 'out'}
        orientation of degrees: full-degree, in-degree or out-degree
    
    Returns
    -------
    list of integers
        union list of networks degrees

    '''
    deg_func = lambda G: {'all': G.degree, 'in': G.in_degree, 'out': G.out_degree}
    return [d for G, coords in networks for n, d in deg_func(G)[deg_type]()]

def plot_distribution(x, y, fmt='o', plt_type='full', plt_func=plt.loglog, **kwargs):
    '''
    Plot degree distribution: y versus x as lines and/or markers.
    
    The coordinates of the points or line nodes are given by *x*, *y*.
    
    Parameters
    ----------
    plt_type : {'full', 'log', 'cdf'}, default='full'
        some `.plot` parameters for different types of distributions by `.distribution`
    plt_func: func, default=plt.loglog
        `.plot` func, for example plt.plot or plt.loglog
    
    Other Parameters
    ----------------
    x, y, fmt, **kwargs
        all parameters supported by `.plot`;
        see https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html
    
    Returns
    -------
    lines
        a list of `~.Line2D` objects representing the plotted data
        
    '''
    plot_kwargs = {
        'full': {'markersize': 3, 'linewidth': 2, 'alpha': 0.5},
        'log': {'markersize': 5, 'linewidth': 2, 'alpha': 1.},
        'cdf': {'markersize': 2, 'linewidth': 2, 'alpha': 1.}
    }
    res_kwargs = plot_kwargs[plt_type] if plt_type else dict()
    res_kwargs.update(kwargs)
    return plt_func(x, y, fmt, **res_kwargs)
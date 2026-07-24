import matplotlib.pyplot as plt

eight_bit_style = {
    # lines
    'lines.linewidth': 3,
    'lines.solid_capstyle': 'butt',

    # font
    'font.family': 'monospace',
    'font.size': 10.0,

    # axes
    'axes.facecolor': 'black',
    'axes.edgecolor': 'white',
    'axes.linewidth': 2,
    'axes.grid': False,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'axes.labelcolor': 'white',
    'axes.spines.left': True,
    'axes.spines.right': True,
    'axes.spines.top': True,
    'axes.spines.bottom': True,

    # xticks
    'xtick.color': 'white',
    'xtick.labelsize': 8,

    # yticks
    'ytick.color': 'white',
    'ytick.labelsize': 8,

    # legend
    'legend.fontsize': 8,
    'legend.frameon': False,
    'legend.loc': 'best',
    'legend.fancybox': False,
    'legend.framealpha': 0.0,

    # text
    'text.color': 'white',

    # figure
    'figure.facecolor': 'black',
    'figure.edgecolor': 'black',

    # savefig
    'savefig.facecolor': 'black',
    'savefig.edgecolor': 'black',

    # Corrected color cycle definition
    'axes.prop_cycle': plt.cycler(color=['#FF5555', '#55FF55', '#5555FF', '#FFFF55', '#FF55FF', '#55FFFF'])
}
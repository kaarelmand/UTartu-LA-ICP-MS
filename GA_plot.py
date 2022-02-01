import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from GA_transform import get_tailing_zero
from math import ceil


def plotwithcol(
    data, x, y, hue=None,
    scatter=True,
    hist=True,
    kde=False,
    truncate=False,
    figsize=(5, 5),
    **kwargs
):
    if truncate:
        data = data.copy()
        data[x].apply()

    fig, ax = plt.subplots(figsize=figsize)
    if scatter:
        sns.scatterplot(
            ax=ax, data=data, x=x, y=y, hue=hue, color=".25",
            linewidth=0, s=5, norm=LogNorm(), **kwargs
        )
    if hist:
        sns.histplot(
            ax=ax, data=data, x=x, y=y, bins=200, pthresh=.1, cmap="mako"
        )
    if kde:
        sns.kdeplot(
            ax=ax, data=data, x='Ca', y='P', color="w", linewidths=1
        )
    return fig, ax


def gridplot(
    data,
    y_vars=['V', 'U'],
    x_vars=['Al', 'C13 cps', 'K', 'Ti49', 'Mg', 'Ca', 'P', 'S'],
    scatter_kwargs=dict(
        linewidth=0, s=2, markers=dict(color=0.15), color='0.15'
    ),
    hist_kwargs=dict(
        bins=200, pthresh=.1, cmap="plasma"
    ),
):
    grid = sns.PairGrid(data=data, y_vars=y_vars, x_vars=x_vars)
    grid.map(sns.scatterplot, **scatter_kwargs)
    grid.map(sns.histplot, **hist_kwargs)
    return grid


def display_cutoffs(data, figscale=2, row_width=6):
    """
    Displays the cutoffs for removing tailing 0-values.
    `figscale` controls the scaling of the figures.
    `row_width` controls how many elements to show on a single row.
    """
    ncols = min([len(data.columns), row_width])
    nrows = ceil(len(data.columns) / row_width)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols*figscale, nrows*figscale)
    )
    for ax, el in zip(axes.flat, data.columns):
        sns.histplot(data[el], ax=ax)
        ax.set_xlabel(el)
        ax.axvline(get_tailing_zero(data[el]), color='red')
        ax.set_yticks([])
        ax.set_ylabel("")

    if nrows > 1:
        emptyaxes = len(axes.flat) - len(data.columns)
        for ax in axes.flat[-emptyaxes:]:
            ax.set_axis_off()

    plt.tight_layout()
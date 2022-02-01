from math import ceil
from string import ascii_lowercase
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pca import pca


def sklearn_pca(
    df,
    n_components=4,
    random_state=1,
    pca_labels=('PC1', 'PC2', 'PC3', 'PC4'),
    export=True,
    exportname="pca_output.xlsx",
):
    # Remove empty values
    input_arr = df.dropna()

    # Scale values
    scaler = StandardScaler()
    scaler.fit(input_arr)
    scaled_arr = scaler.transform(input_arr)

    # Fit PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(scaled_arr)

    # Generate PCA values
    values_arr = pca.transform(scaled_arr)
    values = pd.DataFrame(values_arr, columns=pca_labels, index=input_arr.index)

    # Generate Loadings
    loadings = pd.DataFrame(pca.components_.T, columns=pca_labels, index=df.columns)

    # Generate contributions
    contribs = pd.Series(pca.explained_variance_ratio_.T, index=pca_labels)

    # Export to file
    if export:
        with pd.ExcelWriter(exportname) as writer:
            values.to_excel(writer, sheet_name='sample values')
            loadings.to_excel(writer, sheet_name='variable loadings')
            contribs.to_excel(writer, sheet_name='PC contributions')

    return values, loadings, contribs


def generate_pca(
    df, n_components=4, normalize=True,
    pcname=('PC1', 'PC2', 'PC3', 'PC4'),
    export=True, exportname='pca_export.xlsx',
    **kwargs
):
    # Perform the pca.
    df_nonna = df.dropna()
    model = pca(n_components=n_components, normalize=normalize, **kwargs)
    results = model.fit_transform(df_nonna)

    # Separate values, add outlier info.
    values = results['PC']
    outliers = results['outliers'][['y_bool', 'y_bool_spe']]
    values["Outlier"] = outliers.loc[:, ['y_bool', 'y_bool_spe']].any(axis='columns')

    # Transpose loadings to get a standardized form.
    loadings = results['loadings'].T

    # Contribs are cumulative here; must convert to individual.
    contribscu = results['explained_var']
    contribsarr = [contribscu[0]]
    for i, cont in enumerate(contribscu):
        if i>0:
            contribsarr.append(cont - contribscu[i-1])
    contribs = pd.Series(contribsarr, index=pcname)

    # Export to file
    if export:
        with pd.ExcelWriter(exportname) as writer:
            values.to_excel(writer, sheet_name='PC values and outliers')
            loadings.to_excel(writer, sheet_name='Variable loadings')
            contribs.to_excel(writer, sheet_name='PC contributions')

    return values, loadings, contribs


def label_pca_axes(ax, comps, contrib):
    ax.set_xlabel("{0} ({1:.0f}%)".format(comps[0], contrib[0]*100))
    ax.set_ylabel("{0} ({1:.0f}%)".format(comps[1], contrib[1]*100))


def grid_pca_axes(ax, grid_kwargs):
    if not grid_kwargs:
        grid_kwargs = dict(ls='--',color='0.85')
    ax.axvline(x=0, zorder=0.5, **grid_kwargs)
    ax.axhline(y=0, zorder=0.5, **grid_kwargs)


def pca_scatter(
    ax, values, contrib, comps=('PC1', 'PC2'),
    hue=None, style=None, size=None,
    reflines=True, labels=True,
    scatter_kwargs=None, grid_kwargs=None,
):
    # Default formatting for scatter.
    if not scatter_kwargs:
        scatter_kwargs = dict(color=sns.color_palette()[4], linewidth=0)
    # Scatter the data.
    sns.scatterplot(
        ax=ax,
        data=values, x=comps[0], y=comps[1],
        hue=hue, style=style, size=size,
        **scatter_kwargs,
    )
    # Reference lines and labels.
    if reflines:
        grid_pca_axes(ax, grid_kwargs)
    if labels:
        label_pca_axes(ax, comps, contrib)

    return ax


def pca_arrows(
    ax, loadings, contrib, comps=('PC1', 'PC2'),
    palette=sns.color_palette(),
    reflines=True, labels=True,
    label_offset=0.02,
    arrow_kwargs=None, text_kwargs=None, grid_kwargs=None,
):
    # Make the palette sufficient size.
    if len(palette) < len(loadings):
        mult = ceil(len(loadings)/len(palette))
        palette = palette*mult
    # Default formatting for arrows and labels.
    if not arrow_kwargs:
        arrow_kwargs = dict(head_width=0.01, head_length=0.01)
    if not text_kwargs:
        text_kwargs = dict()
    # Draw arrows from origin and ending at variable loadings.
    for (i, line), color in zip(loadings.iterrows(), palette):
        ax.arrow(
            0, 0, line[comps[0]], line[comps[1]],
            color=color, **arrow_kwargs,
        )
        # Sets text offsets based on quadrant.
        x_offset = label_offset if line[comps[0]] >= 0 else -label_offset
        y_offset = label_offset if line[comps[1]] >= 0 else -label_offset
        ax.text(
            s=i, x=line[comps[0]] + x_offset, y=line[comps[1]] + y_offset,
            color=color, ha='center', va='center', **text_kwargs
        )
    # Reference lines and labels.
    if reflines:
        grid_pca_axes(ax, grid_kwargs)
    if labels:
        label_pca_axes(ax, comps, contrib)


def label_axes(axes):
    for ax, label in zip(axes, ascii_lowercase):
        ax.text(
            s=label, transform=ax.transAxes,
            x=0.05, y=0.95,
            va='top', ha='left', fontweight='bold',
        )

    
def plot_pca(
    values, loadings, contribs,
    pcacomps=(('PC1', 'PC2'), ('PC3', 'PC4')),
    scatter_hue=None, scatter_style=None, scatter_size=None,
    figscale=5, ax_labels=True,
    scatter_kwargs=None, grid_kwargs=None,
):
    # Canvas
    pcafig, pcaaxes = plt.subplots(
        ncols=2, nrows=len(pcacomps), figsize=(figscale*2, figscale*len(pcacomps))
    )
    for axes, comps in zip(pcaaxes, pcacomps):
        # Panel 1
        pca_scatter(
            axes[0], values, comps=comps,
            contrib=contribs.loc[[comps[0], comps[1]]].to_list(),
            hue=scatter_hue, style=scatter_style, size=scatter_size,
            scatter_kwargs=scatter_kwargs, grid_kwargs=grid_kwargs,
        )
        # Panel 2
        pca_arrows(
            axes[1], loadings, comps=comps,
            contrib=contribs.loc[[comps[0], comps[1]]].to_list(),
            arrow_kwargs=None, text_kwargs=None, grid_kwargs=None,
        )
    # Legend
    for ax in pcaaxes.flat[1:]:
        try:
            ax.get_legend().remove()
        except AttributeError:
            pass
    # Labels
    if ax_labels:
        label_axes(pcaaxes.flat)

    return pcafig, pcaaxes


def plot_double_pca(
    values1, loadings1, contribs1,
    values2, loadings2, contribs2,
    pcacomps=(('PC1', 'PC2'), ('PC3', 'PC4')),
    titles=None,
    scatter_hue=None, scatter_style=None, scatter_size=None,
    figscale=5, ax_labels=True,
    scatter_kwargs=None, grid_kwargs=None,
):
    # Canvas
    pcafig = plt.figure(figsize=(figscale*4.2, figscale*len(pcacomps)))
    # Make two grids, both with enough axes to fit
    g1 = GridSpec(ncols=2, nrows=len(pcacomps), right=0.47)
    axes1 = np.empty((2, len(pcacomps)), dtype=Axes)
    for i in range(len(axes1.flat)):
        axes1.flat[i] = pcafig.add_subplot(g1[i])
    g2 = GridSpec(ncols=2, nrows=len(pcacomps), left=0.53)
    axes2 = np.empty((2, len(pcacomps)), dtype=Axes)
    for i in range(len(axes2.flat)):
        axes2.flat[i] = pcafig.add_subplot(g2[i])

    # Draw
    for pcaaxes, (values, loadings, contribs) in zip(
        (axes1, axes2), ((values1, loadings1, contribs1), (values2, loadings2, contribs2))):
        for axes, comps in zip(pcaaxes, pcacomps):
            # Panel 1
            pca_scatter(
                axes[0], values, comps=comps, contrib=contribs.loc[[comps[0], comps[1]]].to_list(),
                hue=scatter_hue, style=scatter_style, size=scatter_size,
                scatter_kwargs=scatter_kwargs, grid_kwargs=grid_kwargs,
            )
            # Panel 2
            pca_arrows(
                axes[1], loadings, comps=comps, contrib=contribs.loc[[comps[0], comps[1]]].to_list(),
                arrow_kwargs=None, text_kwargs=None, grid_kwargs=None,
            )

    # Titles
    if titles:
        axes1.flat[0].text(
            s=titles[0], x=0, y=1.02, transform=axes1.flat[0].transAxes,
            ha='left', va='bottom', fontsize='x-large'
        )
        axes2.flat[1].text(
            s=titles[1], x=1, y=1.02, transform=axes2.flat[1].transAxes,
            ha='right', va='bottom', fontsize='x-large'
        )
    # Legend
    for ax in pcafig.axes[1:]:
        try:
            ax.get_legend().remove()
        except AttributeError:
            pass
    # Labels
    if ax_labels:
        label_axes(pcafig.axes)

    return pcafig


def full_outlier_pca(df, random_state=1, export=True, outname='pca'):
    # Run pca for the first time
    values, loadings, contribs = generate_pca(
        df, random_state=random_state, export=True,
        exportname=f'{outname}_with_outliers.xlsx'
    )
    # Remove outliers
    df_noout = df.dropna()[~values['Outlier']]
    values2, loadings2, contribs2 = generate_pca(
        df_noout, random_state=1, export=True,
        exportname=f'{outname}_without_outliers.xlsx'
    )

    fig = plot_double_pca(
        values, loadings, contribs,
        values2, loadings2, contribs2,
        titles=('With outliers', 'Outliers removed'),
        scatter_hue='Outlier', scatter_size='Outlier',
        scatter_kwargs=dict(
            palette=(sns.color_palette()[4], '0.5'), 
            sizes=(5, 10), linewidth=0,
        )
    )

    outdict = {'values': values, 'loadings': loadings, 'contribs': contribs}
    nooutdict = {'values': values2, 'loadings': loadings2, 'contribs': contribs2}

    return fig, outdict, nooutdict
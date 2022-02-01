import numpy as np
import pandas as pd
from scipy import stats


def laicpms_melt(df, name):
    """
    Melt a single LA-ICP-MS spatial dataframe to a long-form one.
    """
    df = df.copy()
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={'index': 'y'}, inplace=True)
    df = df.melt(
        id_vars=['y'], value_vars=df.columns[1:], var_name='x', value_name=name
    )
    return df


def join_laicpms_elements(filename, export=True, exportname=False):
    """
    Takes a filename containing LA-ICP-MS data and joins them into a wide-form
    table with each element given as a column. `export` allows saving to an
    excel sheet with a default name being {filename}_combined. `exportname`
    allows overriding the default.
    """
    # Read LAICPMS file into separate sheets.
    input_dict = pd.read_excel(filename, sheet_name=None, header=None)
    # Melt each sheet into a long-form table.
    molten_list = [
        laicpms_melt(df, element) for element, df in input_dict.items()
    ]
    # Concatenate all separate long-form dataframes to a single wide-form
    # dataframe with measured elements as columns.
    df = pd.concat(molten_list, axis=1)
    # Remove duplicate x and y columns.
    df = df.loc[:, ~df.columns.duplicated()]

    # Save to file.
    if export:
        if exportname:
            df.to_excel(exportname, index=False)
        else:
            df.to_excel(f"{filename}_combined.xlsx")

    return df


def drop_outliers(data, z_thres=4):
    """
    Return dataframe where outliers are replaced by nan on a per-column basis.
    Outliers are calculated based on a z-score threshold (`z_thres`).
    """
    zdata = stats.zscore(data, axis=0, nan_policy='omit')
    return data.where(abs(zdata) < z_thres)


def drop_zero(data):
    """
    Remove zero values from dataframe.
    """
    return data.where(data > 0)


def get_tailing_zero(series, bins=100, meancutoff=2000):
    """
    In a `series` of LA-ICP-MS data, finds the value below which superfluous
    "tailing" 0-values abound. Fidelity depends on the `bins` parameter.
    `meancutoff` is the maximum series mean value at which the returned real
    zero is simply the inimum of the series (low mean values do not have tailing
    zeros Set to `False` to disable.
    """
    # Don't work with low-mean series.
    if meancutoff and series.mean() < meancutoff:
        return series.min()

    # Consider only low values below the 25% quantile.
    lowseries = series[series < series.quantile(0.25)]

    # Bin the samples.
    bin_bounds = np.linspace(lowseries.min(), lowseries.max(), num=bins)
    binned = pd.cut(
        lowseries, bin_bounds, labels=bin_bounds[1:]
    ).astype(np.float64)

    # Find concentration at which counts are minimal.
    v_counts = binned.value_counts()
    real_zero = v_counts.idxmin()

    return real_zero


def drop_tailing_zero(data):
    """
    Remove "tailing" near-zero values from dataframe.
    """
    data = data.copy()
    for el in data.columns:
        data[el] = data[el].where(data[el] > get_tailing_zero(data[el]))
    return data


def clean_data(data):
    return drop_tailing_zero(drop_zero(drop_outliers(data)))
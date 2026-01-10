from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, root_mean_squared_error


def print_best_models(results: Dict[str, Dict[str, Any]], model: str) -> None:
    """
    Prints the best model's score and parameters from the results of a hyperparameter search.

    Parameters:
    results (Dict[str, Dict[str, Any]]): A dictionary containing the results of the model search,
                                          where each key is a model name and the value is another
                                          dictionary with model evaluation metrics.
    model (str): The name of the model for which to print the best score and parameters.

    Returns:
    None: This function does not return a value; it prints the results directly.
    """
    model_results = results[model]
    id = model_results['rank_test_score'].argmin()

    score = model_results['mean_test_score'][id]*-1
    params = model_results['params'][id]

    print(f"""Model: {model} \nBest score: {score:.2f} RMSE \nBest parameters: {params} \n""")


def plot_predictions(
        y_true: pd.Series,
        y_pred: np.ndarray,
        title: str = 'Model Predictions vs. Ground Truth',
        color='#1eb49c') -> None:
    """
    Plots the model predictions against the ground truth values.

    Parameters:
    y_true (pd.Series): The ground truth values.
    y_pred (np.ndarray): The predicted values from the model.
    title (str): The title of the plot. Default is 'Decision Tree: Model Predictions vs. Ground Truth'.
    color (str): The color of the plot. Default is '#1eb49c'.

    Returns:
    None
    """
    # Calculate R^2 and mean squared error
    r2 = r2_score(y_true, y_pred)
    mse = root_mean_squared_error(y_true, y_pred)

    # Create scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, color=color)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title(title)

    # Add R^2 and root mean squared error to the plot with thousand separator
    plt.text(0.05, 0.95, f'R^2: {r2:.2f}\nRMSE: {mse:,.2f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Add thousand separator to the axis labels
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{int(x):,}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, loc: f'{int(y):,}'))

    plt.show()


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')


class DataFrameSimpleImputer(BaseEstimator, TransformerMixin):
    """
    A custom imputer that wraps around SimpleImputer to ensure that the output is a pandas DataFrame.
    """

    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)

    def fit(self, X: pd.DataFrame, y=None) -> 'DataFrameSimpleImputer':
        """
        Fits the imputer on the DataFrame.
        """
        self.imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame by imputing missing values and returns a DataFrame.
        """
        # Perform the imputation
        imputed_array = self.imputer.transform(X)

        # Create a DataFrame with the same columns as the original DataFrame
        return pd.DataFrame(imputed_array, columns=X.columns, index=X.index)


class VINReplacer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for replacing missing values in the 'year' and 'manufacturer' columns
    based on the VIN (Vehicle Identification Number) column.

    This transformer uses predefined mappings to fill in missing values:
    - 'year' is filled using the 10th character of the VIN mapped to a year.
    - 'manufacturer' is filled using the first three characters of the VIN mapped to a manufacturer.

    Attributes:
        vin_to_year (dict): A dictionary mapping VIN characters to corresponding year values.
        vin_to_manufacturer (dict): A dictionary mapping VIN characters to corresponding manufacturer values.
    """

    def __init__(self, vin_to_year: dict, vin_to_manufacturer: dict) -> None:
        """
        Initializes the VINValueReplacer with mappings for year and manufacturer.

        Args:
            vin_to_year (dict): Mapping from VIN characters to year values.
            vin_to_manufacturer (dict): Mapping from VIN characters to manufacturer values.
        """
        self.vin_to_year = vin_to_year
        self.vin_to_manufacturer = vin_to_manufacturer

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'VINReplacer':
        """
        Fits the transformer to the data. This transformer does not require fitting.

        Args:
            X (pd.DataFrame): The input data.
            y (pd.Series, optional): The target values (not used).

        Returns:
            VINReplacer: The fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input data by replacing missing values in 'year' and 'manufacturer'.

        Args:
            X (pd.DataFrame): The input data with potential missing values.

        Returns:
            pd.DataFrame: The transformed data with missing values replaced.
        """
        # Replace missing 'year' values
        missing_years = X.year.isnull()
        X.loc[missing_years, 'year'] = X.loc[missing_years].VIN.apply(
            lambda x: x[9]
        ).map(self.vin_to_year)

        # Replace missing 'manufacturer' values
        missing_manufacturers = X.manufacturer.isnull() & X.VIN.notnull()
        X.loc[missing_manufacturers, 'manufacturer'] = X.loc[missing_manufacturers].VIN.apply(
            lambda x: x[0:3]
        ).map(self.vin_to_manufacturer)

        return X


class ConditionalImputer(BaseEstimator, TransformerMixin):
    """
    A custom imputer that fills missing values in a target column based on the most frequent
    or mean value conditioned on one or more other columns.

    Parameters:
    ----------
    target_col : str
        The name of the column to impute.
    condition_cols : list[str]
        A list of column names to condition the imputation on.
    strategy : str, default='most_frequent'
        The imputation strategy to use. Can be 'most_frequent' or 'mean'.

    Methods:
    -------
    fit(X, y=None):
        Learns the imputation values based on the specified strategy.
    transform(X):
        Imputes missing values in the target column based on the learned values.
    """

    def __init__(self, target_col: str, condition_cols: list[str], strategy: str = 'most_frequent'):
        self.strategy = strategy
        self.condition_cols = condition_cols
        self.target_col = target_col
        self.impute_values = {}

    def fit(self, X: pd.DataFrame, y=None) -> 'ConditionalImputer':
        """
        Learns the imputation values based on the specified strategy.

        Parameters:
        ----------
        X : pd.DataFrame
            The input data containing the target and condition columns.
        y : None
            Ignored. This parameter is included for compatibility with scikit-learn.

        Returns:
        -------
        self : ConditionalImputer
            Fitted imputer instance.
        """

        if self.strategy == 'most_frequent':
            self.impute_values = (
                X.groupby(self.condition_cols)[self.target_col]
                .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
                .to_dict()
            )
        elif self.strategy == 'mean':
            self.impute_values = (
                X.groupby(self.condition_cols)[self.target_col]
                .agg(lambda x: np.nanmean(x))
                .to_dict()
            )
        else:
            raise ValueError('Invalid strategy')
        return self

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """
        Imputes missing values in the target column based on the learned values and returns the entire column.

        Parameters:
        ----------
        X : pd.DataFrame
            The input data containing the target and condition columns.

        Returns:
        -------
        pd.Series
            The target column with missing values filled.
        """

        # Create a copy to avoid modifying the original DataFrame
        imputed_column = X[self.target_col].copy()

        # Iterate over the groups and their corresponding impute values
        for group, value in self.impute_values.items():
            # Create a boolean mask for all condition columns
            if len(self.condition_cols) == 1:
                mask = (X[self.condition_cols[0]] == group)
            else:
                mask = (X[self.condition_cols] == group).all(axis=1)

            # Fill in the imputed values where the target column is null
            imputed_column.loc[mask & imputed_column.isnull()] = value

        X.loc[:, self.target_col] = imputed_column

        return X


class SportColumn(BaseEstimator, TransformerMixin):
    """
    A custom transformer that calculates a sports package indicator from the model description.
    """

    def fit(self, X: pd.DataFrame, y=None) -> 'SportColumn':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Calculate the 'age' column
        X['sport'] = X['model'].apply(lambda x: 'sport' in x.lower()).astype(int)
        return X


class AgeCalculator(BaseEstimator, TransformerMixin):
    """
    A custom transformer that calculates the 'age' column based on 'posting_date' and 'year'.
    """

    def fit(self, X: pd.DataFrame, y=None) -> 'AgeCalculator':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Calculate the 'age' column
        try:
            sales_year = X['posting_date'].dt.year
        except AttributeError:
            sales_year = pd.to_datetime(X['posting_date']).dt.year
        X['age'] = sales_year - X['year']
        return X


def plot_univariate(
        df,
        columns,
        hue=None,
        bins=50,
        bw_method=0.1,
        size=(20, 24),
        ncols=2,
        hspace=0.7,
        wspace=0.2,
        log_dict=None,
        color='#1eb49c'):
    """Function to visualize columns in df. Visualization type depends on data type of the column.

    Arguments
    ---------
    df : pandas.DataFrame
        Dataframe whose columns shall be visualized.
    columns : list
        Subset of columns which shall be considered only.
    hue: str
        Column according to which single visualization shall be grouped.
    bins : int
        Number of bins for the histogram plots.
    bw_method : float
        method for determining the smoothing bandwidth to use.
    size: tuple
        Size of the resulting grid.
    nclos: int
        Number of columns in the resulting grid.
    hspace : float
        Horizontal space between subplots.
    wspace : float
        Vertical space between subplots.
    log_dict : dict
        Dictionary listing whether a column's visualization should be
        displayed in log scale on the vertical axis
    color : str
        Color used in the plots.


    Returns
    -------
    Visualization of each variable in columns as barplot or histogram.

    """

    # Reduce df to relevant columns
    df = df[columns]

    # Calculate the number of rows and columns for the grid
    num_cols = len(df.columns)
    num_rows = int(num_cols / ncols) + (num_cols % ncols)

    # Create the subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=ncols, figsize=size)

    # Change the vertical and horizontal spacing between subplots
    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Do not display vertical axis in log scale as default
    logy = False

    # Iterate over each column and plot accordingly
    for i, column in enumerate(df.columns):
        if log_dict is not None:
            logy = log_dict.get(column, False)

        ax = axes[i]
        # Barplots for categorical features or integers with few distinct values
        if (
            (df[column].dtype == 'int64' and df[column].value_counts().shape[0] < 40) or
            df[column].dtype == 'object' or
            df[column].dtype == '<M8[ns]'
        ):
            if hue is None or hue == column:
                temp = df[column].value_counts().sort_index()
                if temp.shape[0] > 20:
                    fontsize = 'small'
                else:
                    fontsize = 'medium'
                temp.plot(
                    kind='bar',
                    ax=ax,
                    ylabel='Count',
                    xlabel='',
                    title=column,
                    logy=logy,
                    fontsize=fontsize,
                    color=color)
            else:
                temp = df[[column, hue]].groupby(hue).value_counts(normalize=True).sort_index().to_frame().reset_index()
                if temp.shape[0] > 20:
                    fontsize = 'small'
                else:
                    fontsize = 'medium'
                temp[hue] = temp[hue].astype(str)
                p = sns.barplot(temp, x=column, y='proportion', hue=hue, errorbar=None, ax=ax, color=color)
                # Add title and labels
                p.set_title(column)
                p.set_xlabel('')
                p.set_ylabel('Proportion')
                p.set_xticks(p.get_xticks())
                p.set_xticklabels(
                    p.get_xticklabels(),
                    rotation=90,
                    horizontalalignment='center',
                    fontsize=fontsize)
                if logy:
                    p.set_yscale("log")

        # Histograms for floats or integers with many distinct values
        elif (
            df[column].dtype == 'int64' and df[column].value_counts().shape[0] >= 10 or
            df[column].dtype == 'float64'
        ):
            if hue is None:
                df[column].plot(kind='hist', ax=ax, bins=bins, title=column, logy=logy, color=color)
            else:
                hue_groups = np.sort(df[hue].unique())
                for hue_group in hue_groups:
                    p = sns.kdeplot(
                        data=df[df[hue] == hue_group],
                        x=column,
                        fill=True,
                        label=hue_group,
                        ax=ax,
                        bw_method=bw_method,
                        color=color)
                # Add title and labels
                p.set_title(column)
                p.set_xlabel('')
                p.set_ylabel('Density')
                p.legend(title=hue)
                if logy:
                    p.set_yscale("log")

        # For all other data types pass
        else:
            pass


def plot_time_series(
        df: pd.DataFrame,
        x: str,
        y_primary: list[str],
        y_secondary: list[str] = None,
        title: str = None,
        xlabel: str = None,
        ylabel_primary: str = None,
        ylabel_secondary: str = None,
        figsize=(10, 6),
        nbins=5) -> None:
    """Function to plot time series data.

    Arguments
    ---------
    df : pd.DataFrame
        DataFrame containing the time series data.
    x : str
        Column name to be used for the x-axis (typically the date or time column).
    y_primary : list[str]
        List of column names to be plotted on the primary y-axis.
    y_secondary : list[str], optional
        List of column names to be plotted on the secondary y-axis. Default is None.
    title : str, optional
        Title of the plot. Default is None.
    xlabel : str, optional
        Label for the x-axis. Default is None.
    ylabel_primary : str, optional
        Label for the primary y-axis. Default is None.
    ylabel_secondary : str, optional
        Label for the secondary y-axis. Default is None.
    figsize : tuple, optional
        Size of the figure. Default is (10, 6).
    nbins : int, optional
        Number of bins for the x-axis major locator. Default is 5.
        This function does not return any value. It displays a plot of the time series data.

    Returns
    -------
    Visualization of time series.

    """

    fig, ax1 = plt.subplots(figsize=figsize)

    primary_colors = plt.cm.tab10.colors[:len(y_primary)]

    # Plot the primary y-axis
    for i, y in enumerate(y_primary):
        ax1.plot(df[x], df[y], label=y, color=primary_colors[i])
    ax1.set_xlabel('Date')
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax1.set_ylabel(ylabel_primary)
    ax1.set_xlabel(xlabel)
    ax1.legend(loc='upper left')
    ax1.yaxis.grid(True, linestyle='--')

    # Plot the secondary y-axis
    if y_secondary is not None:
        secondary_colors = plt.cm.tab10.colors[len(y_primary):len(y_primary) + len(y_secondary)]
        ax2 = ax1.twinx()
        for i, y in enumerate(y_secondary):
            ax2.plot(df[x], df[y], label=y, color=secondary_colors[i])
        ax2.set_ylabel(ylabel_secondary)
        ax2.legend(loc='upper right')

    plt.title(title)
    fig.autofmt_xdate()
    plt.show()


vin_to_year = {
    'T': 1996,
    'V': 1997,
    'W': 1998,
    'X': 1999,
    'Y': 2000,
    '1': 2001,
    '2': 2002,
    '3': 2003,
    '4': 2004,
    '5': 2005,
    '6': 2006,
    '7': 2007,
    '8': 2008,
    '9': 2009,
    'A': 2010,
    'B': 2011,
    'C': 2012,
    'D': 2013,
    'E': 2014,
    'F': 2015,
    'G': 2016,
    'H': 2017,
    'J': 2018,
    'K': 2019,
    'L': 2020,
    'M': 2021,
    'N': 2022,
    'P': 2023,
    'R': 2024,
    'S': 2025
    }


vin_to_manufacturer = {
    '19U': 'acura',
    '1B3': 'dodge',
    '1B4': 'dodge',
    '1B7': 'dodge',
    '1C3': 'chrysler',
    '1C4': 'chrysler',
    '1C6': 'chrysler',
    '1D3': 'dodge',
    '1D4': 'dodge',
    '1D7': 'dodge',
    '1D8': 'dodge',
    '1FA': 'ford',
    '1FB': 'ford',
    '1FC': 'ford',
    '1FD': 'ford',
    '1FE': 'ford',
    '1FF': 'ford',
    '1FG': 'ford',
    '1FH': 'ford',
    '1FM': 'ford',
    '1FT': 'ford',
    '1FU': 'freightliner',
    '1FV': 'freightliner',
    '1FW': 'ford',
    '1FX': 'ford',
    '1FY': 'ford',
    '1FZ': 'ford',
    '1G1': 'chevrolet',
    '1G6': 'cadillac',
    '1GC': 'chevrolet trucks',
    '1GD': 'chevrolet',
    '1GM': 'pontiac',
    '1GT': 'gmc',
    '1HD': 'international',
    '1HG': 'honda',
    '1N4': 'nissan',
    '1N6': 'nissan',
    '2A4': 'chrysler',
    '2A8': 'chrysler',
    '2B3': 'dodge',
    '2B4': 'dodge',
    '2B7': 'dodge',
    '2C3': 'chrysler',
    '2C4': 'chrysler',
    '2C8': 'chrysler',
    '2D4': 'dodge',
    '2D8': 'dodge',
    '2FU': 'freightliner',
    '2FV': 'freightliner',
    '2G1': 'chevrolet',
    '2G2': 'pontiac',
    '2G3': 'oldsmobile',
    '2G4': 'oldsmobile',
    '2HG': 'honda',
    '2HK': 'honda',
    '2HM': 'honda',
    '2T1': 'toyota',
    '2T2': 'toyota',
    '2T3': 'toyota',
    '3A1': 'freightliner',
    '3A6': 'kenworth',
    '3A7': 'peterbilt',
    '3FA': 'ford',
    '3G1': 'chevrolet',
    '3G3': 'oldsmobile',
    '3G4': 'oldsmobile',
    '3G5': 'buick',
    '3GK': 'freightliner',
    '3HG': 'honda',
    '3HS': 'international',
    '3N1': 'nissan',
    '3VW': 'volkswagen',
    '4F2': 'mazda',
    '4F4': 'mazda',
    '4M2': 'mercury',
    '4S3': 'subaru',
    '4S4': 'subaru',
    '4S6': 'subaru',
    '4T1': 'toyota',
    '4T3': 'toyota',
    '4T4': 'toyota',
    '5J6': 'honda',
    '5J8': 'honda',
    '5KM': 'hyosung',
    '5N1': 'nissan',
    '5NP': 'hyundai',
    '5TD': 'toyota',
    '5TE': 'toyota',
    '5TF': 'toyota',
    '5YJ': 'tesla',
    '6F5': 'ford australia',
    '6FP': 'ford australia',
    '6G1': 'holden',
    '6G2': 'pontiac australia',
    '6H8': 'holden',
    '6T9': 'holden',
    '7G2': 'tesla',
    '7SA': 'tesla',
    '8A1': 'renault argentina',
    '8AB': 'ford argentina',
    '8AC': 'renault argentina',
    '8AD': 'peugeot argentina',
    '8AE': 'chevrolet argentina',
    '8AF': 'ford argentina',
    '8AG': 'chevrolet argentina',
    '8AJ': 'toyota argentina',
    '8AP': 'fiat argentina',
    '936': 'peugeot brazil',
    '937': 'fiat brazil',
    '938': 'fiat brazil',
    '939': 'fiat brazil',
    '93H': 'honda brazil',
    '93U': 'audi brazil',
    '93X': 'mitsubishi brazil',
    '93Y': 'renault brazil',
    '9BD': 'fiat brazil',
    '9BE': 'renault brazil',
    '9BF': 'ford brazil',
    '9BG': 'chevrolet brazil',
    '9BH': 'hyundai brazil',
    '9BM': 'mercedes benz brazil',
    '9BR': 'toyota brazil',
    '9BS': 'scania brazil',
    '9BV': 'volvo brazil',
    '9BW': 'volkswagen brazil',
    'AAV': 'volkswagen south africa',
    'AFA': 'ford south africa',
    'AHT': 'toyota south africa',
    'CL9': 'fiat tunisia',
    'JA3': 'mitsubishi',
    'JA4': 'mitsubishi',
    'JF1': 'subaru',
    'JF2': 'subaru',
    'JH4': 'acura',
    'JH6': 'acura',
    'JHL': 'honda',
    'JHM': 'honda',
    'JKA': 'kawasaki',
    'JM1': 'mazda',
    'JM3': 'mazda',
    'JM7': 'mazda',
    'JMB': 'mitsubishi',
    'JN1': 'nissan',
    'JN6': 'nissan',
    'JN8': 'nissan',
    'JNA': 'nissan',
    'JNC': 'nissan',
    'JNK': 'nissan',
    'JT2': 'toyota',
    'JT3': 'toyota',
    'JT4': 'toyota',
    'JT5': 'toyota',
    'JT6': 'toyota',
    'JT8': 'toyota',
    'JTA': 'toyota',
    'JTB': 'toyota',
    'JTC': 'toyota',
    'JTD': 'toyota',
    'JTE': 'toyota',
    'JTF': 'toyota',
    'JTG': 'toyota',
    'JTH': 'toyota',
    'JTJ': 'toyota',
    'JTK': 'toyota',
    'JTL': 'toyota',
    'JTM': 'toyota',
    'JTN': 'toyota',
    'JYA': 'yamaha',
    'KL1': 'daewoo',
    'KL2': 'daewoo',
    'KL4': 'daewoo',
    'KMH': 'hyundai',
    'KMJ': 'hyundai',
    'KNA': 'kia',
    'KNB': 'kia',
    'KNC': 'kia',
    'KND': 'kia',
    'KNE': 'kia',
    'KNF': 'kia',
    'KNG': 'kia',
    'KNH': 'kia',
    'KNJ': 'kia',
    'KNK': 'kia',
    'KNL': 'kia',
    'KNM': 'kia',
    'KNN': 'kia',
    'KNP': 'kia',
    'KNR': 'kia',
    'KNS': 'kia',
    'KNT': 'kia',
    'KNU': 'kia',
    'KNV': 'kia',
    'KNW': 'kia',
    'KNX': 'kia',
    'KNY': 'kia',
    'KNZ': 'kia',
    'KPT': 'ssangyong',
    'L15': 'baic bjev',
    'L56': 'byd',
    'LAN': 'changhe',
    'LBE': 'beijing automotive',
    'LBV': 'bmw brilliance',
    'LDC': 'dongfeng motor',
    'LDY': 'zhengzhou yutong',
    'LEN': 'changan',
    'LEV': 'changan',
    'LFG': 'gac',
    'LFM': 'faw',
    'LFP': 'faw',
    'LFV': 'faw',
    'LGB': 'byd',
    'LGE': 'geely',
    'LGH': 'great wall',
    'LGW': 'great wall',
    'LGX': 'beijing automotive',
    'LH1': 'faw haima',
    'LHD': 'beijing hyundai',
    'LHE': 'hyundai',
    'LHG': 'haval',
    'LJD': 'dongfeng peugeot',
    'LKL': 'dongfeng motor',
    'LLA': 'geely',
    'LLV': 'lifan',
    'LMG': 'guangzhou honda',
    'LMP': 'dongfeng motor',
    'LMU': 'xpeng',
    'LNB': 'beijing benz',
    'LNP': 'chery',
    'LPA': 'changan',
    'LPR': 'changan',
    'LRW': 'nio',
    'LSG': 'saic gm',
    'LSH': 'saic',
    'LSJ': 'shanghai gm',
    'LSV': 'saic motor',
    'LTV': 'chery',
    'LUC': 'changan suzuki',
    'LVG': 'gac',
    'LVS': 'li auto',
    'LVV': 'chery',
    'LVY': 'chery',
    'LWV': 'gac toyota',
    'LYV': 'gac',
    'LZE': 'jac motors',
    'LZG': 'gac',
    'LZM': 'dongfeng motor',
    'LZW': 'bmw brilliance',
    'LZY': 'yutong',
    'MA1': 'mahindra',
    'MA3': 'mahindra',
    'MA6': 'mahindra',
    'MA7': 'mahindra',
    'MAL': 'mahindra',
    'MBJ': 'bajaj auto',
    'ME1': 'bajaj',
    'ME4': 'tata motors',
    'MHR': 'hero motocorp',
    'NMT': 'tata motors',
    'RF9': 'suzuki',
    'SAJ': 'jaguar',
    'SAL': 'land rover',
    'SAR': 'jaguar',
    'SAS': 'jaguar',
    'SAT': 'jaguar',
    'SCA': 'rolls royce',
    'SCB': 'bentley',
    'SCC': 'rolls royce',
    'SCE': 'bentley',
    'SDB': 'peugeot uk',
    'SFA': 'aston martin',
    'SFD': 'aston martin',
    'SFZ': 'tesla',
    'SJN': 'mclaren',
    'TMA': 'hyundai',
    'TMB': 'skoda',
    'TMC': 'skoda',
    'TME': 'skoda',
    'TMK': 'skoda',
    'TML': 'skoda',
    'TMM': 'skoda',
    'TMN': 'skoda',
    'TMP': 'skoda',
    'TMT': 'skoda',
    'TMW': 'skoda',
    'TMX': 'skoda',
    'TMY': 'skoda',
    'TMZ': 'skoda',
    'TNE': 'ferrari',
    'TRU': 'audi',
    'VBK': 'ktm',
    'VF1': 'renault',
    'VF3': 'peugeot',
    'VF6': 'renault trucks',
    'VF7': 'citroen',
    'VF8': 'peugeot',
    'VF9': 'peugeot',
    'VFE': 'iveco',
    'VFF': 'iveco',
    'VNE': 'renault',
    'VNK': 'toyota europe',
    'VR1': 'ds automobiles',
    'VSA': 'opel',
    'VSK': 'opel',
    'VSS': 'seat',
    'VSX': 'opel',
    'VSY': 'opel',
    'VSZ': 'opel',
    'W0L': 'opel',
    'W0V': 'opel',
    'WAU': 'audi',
    'WBA': 'bmw',
    'WBS': 'bmw',
    'WBX': 'bmw',
    'WDB': 'mercedes benz trucks',
    'WDC': 'mercedes benz trucks',
    'WDD': 'mercedes-benz',
    'WDF': 'mercedes-benz',
    'WF0': 'ford europe',
    'WMW': 'mini',
    'WP0': 'porsche',
    'WP1': 'porsche',
    'WV1': 'volkswagen',
    'WV2': 'volkswagen',
    'WVG': 'volkswagen',
    'WVW': 'volkswagen',
    'XP7': 'polestar',
    'XTA': 'lada',
    'XTT': 'lada',
    'YK1': 'saab',
    'YS3': 'saab',
    'YS4': 'saab',
    'YTN': 'saab',
    'YV1': 'volvo',
    'YV2': 'volvo trucks',
    'YV3': 'volvo trucks',
    'YV4': 'volvo',
    'ZAM': 'maserati',
    'ZAP': 'aprilia',
    'ZCF': 'iveco',
    'ZD3': 'ducati',
    'ZDC': 'tata motors',
    'ZDM': 'ducati',
    'ZFA': 'fiat',
    'ZFF': 'ferrari',
    'ZHW': 'lamborghini',
}

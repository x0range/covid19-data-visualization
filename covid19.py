""" Script to visualize recent COVID-19 (Coronavirus) infections using data from
    https://github.com/CSSEGISandData/COVID-19/
    Author: Torsten Heinrich
    Date: 2020-03-10
"""

import requests
import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    """
    Function for downloading and loading data. Will download and save csv files and load the data.
    Arguments: None.
    Returns: dict with three DataFrames for total cases, deaths, and recovered patients.
    """
    
    """ Prepare variables"""
    urls = {"cases": "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv",
            "deaths": "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv",
            "recovered": "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv"}

    localnames = {"cases": "Cases.csv",
                  "deaths": "Deaths.csv",
                  "recovered": "Recovered.csv"}

    dfs = {"cases": None,
           "deaths": None,
           "recovered": None}

    if False:
        """ Download"""
        for key in urls.keys():
            url = urls[key]
            localname = localnames[key]
            urllib.request.urlretrieve(url, localname)

    """ Load variables"""
    for key in dfs.keys():
        dfs[key] = pd.read_csv(localnames[key])
    
    """ Return"""
    return(dfs)
    
def preprocess(df, combine_list, single_provinces=["Hubei"]):
    """
    Function for preprocessing DataFrames, dropping unused columns, combining rows for country provinces
    Arguments: 
        df: pandas DataFrame -              The data
        combine_list: list of string -      Country names for which the provinces should be combined
        single_provinces: list of strings - Provinces that should remain separate
    Returns pandas DataFrame.
    """
    
    """ Mark single regions that are to remain separate"""
    for single_province in single_provinces:
        df.loc[df["Province/State"]==single_province, "Country/Region"] = single_province
    
    """ Combine rows for other country provinces"""
    next_index = max(df.index)
    for singlename in combine_list:
        
        """ Select country"""
        singlecountry = df.loc[df["Country/Region"]==singlename,:]
        
        """ Compute sum of provinces"""
        singlesum = singlecountry.sum(axis=0)
        
        """ Set other column variables"""
        singlesum["label"] = singlename
        singlesum["Province/State"] = np.nan
        singlesum["Country/Region"] = singlename
        
        """ Drop provinces from DataFrame"""
        df = df.loc[df["Country/Region"]!=singlename,:]
        
        """Merge country sum into DataFrame"""
        singlesum.name = next_index
        next_index += 1
        df = df.append(singlesum)

    """ Rename rest of Mainland China"""
    df.loc[df["Country/Region"]=="Mainland China", "Country/Region"] = "Mainland China w/o Hubei"
    df.loc[df["Country/Region"]=="China", "Country/Region"] = "China w/o Hubei"
    
    """ Reset index to region name"""
    df["label"] = df["Country/Region"]
    df.loc[pd.notna(df["Province/State"]),"label"] = df.loc[pd.notna(df["Province/State"]),:]["Province/State"]
    df.index = df["label"]
    
    df = df.sort_index()
    """ Drop unused columns"""
    df = df.drop(['Province/State', 'Country/Region', 'Lat', 'Long', "label"], axis = 1) 
    
    """ Return"""
    return df

def compute_deaths_over_closed(dfs):
    """
    Function for computing deaths over colsed cases
    Arguments: 
        dfs: dict of pandas DataFrames - The data
    Returns dict of pandas DataFrames with additionally the dataFrame for deaths_over_closed.
    """
    
    dfcols = dfs["deaths"].columns
    dfs["deaths_over_closed"] = dfs["deaths"].copy()

    for ccol in dfcols:
        dfs["deaths_over_closed"][ccol] = dfs["deaths"][ccol] / (dfs["recovered"][ccol] + dfs["deaths"][ccol])
    
    return(dfs)

def compute_active_cases(dfs):
    """
    Function for computing active cases
    Arguments: 
        dfs: dict of pandas DataFrames - The data
    Returns dict of pandas DataFrames with additionally the dataFrame for active_cases.
    """
    
    dfcols = dfs["cases"].columns
    dfs["active_cases"] = dfs["cases"].copy()

    for ccol in dfcols:
        dfs["active_cases"][ccol] = dfs["cases"][ccol] - dfs["recovered"][ccol] - dfs["deaths"][ccol]
    
    return(dfs)

def compute_death_rate(dfs):
    """
    Function for computing deaths rate
    Arguments: 
        dfs: dict of pandas DataFrames - The data
    Returns dict of pandas DataFrames with additionally the dataFrame for death_rate.
    """
    
    dfcols = dfs["deaths"].columns
    dfs["death_rate"] = dfs["deaths"].copy()

    for ccol in dfcols:
        dfs["death_rate"][ccol] = dfs["deaths"][ccol] / dfs["cases"][ccol]
    
    return(dfs)



def compute_active_cases_reindexed(dfs):
    """
    Function for reindexing active cases so that the start of the epidemic is in the same time period for all countries.
    Arguments: 
        dfs: dict of pandas DataFrames - The data
    Returns dict of pandas DataFrames with additionally the dataFrame for active_cases_reindexed.
    """
    
    """ Prepare data frame, transpose, drop calendar index"""
    df = dfs["active_cases"].copy()
    df = df.T
    df = df.reset_index(drop=True)

    """ Add two time periods (Hubei is otherwise too long or starts too late) """
    for i in range(2):
        df.append(pd.Series(), ignore_index=True)
    
    len_data = len(df)
    
    """ Go through countries, shift start of the epidemic to the beginning of the data frame"""
    dfcols = df.columns
    for ccol in dfcols:
        idx = np.argmax(np.asarray(df[ccol])>=100)
        if idx==0 and df[ccol][0] < 100:
            idx = len_data
        """Denmark and South Korea have big jumps at ~ 100 cases"""
        if ccol in ["Denmark", "Korea, South"]:
            idx -= 1
        """Hubei starts two time periods after start of the epidemic, most other start too early"""
        if ccol != "Hubei":
            replacement_0 = np.asarray(df[ccol][idx:])
            replacement_1 = np.empty(idx)
            replacement_1[:] = np.nan
        else:
            replacement_1 = np.asarray(df[ccol][:-2])
            replacement_0 = np.empty(2)
            replacement_0[:] = np.nan
            
        replacement = np.hstack((replacement_0, replacement_1))
        df[ccol] = pd.Series(replacement)
    
    """ Transpose back, return"""
    dfs["active_cases_reindexed"] = df.T
    return(dfs)
    
def plot(df, row_inclusion_index, title, filename):
    """
    Function for plotting
    Arguments: 
        df: pandas DataFrame -                              The data to be plotted
        row_inclusion_index: pandas DataSeries of bool -    Which rows (regions) to plot. The rest is dropped
        title: string -                                     Plot title
        filename: string -                                  File name used for saving the figure
    Returns dict of pandas DataFrames with additionally the dataFrame for deaths_over_closed.
    """
    
    """ Reduce DataFrame to required rows (regions)"""
    df_reduced = df.loc[row_inclusion_index]
    """ Transpose DataFrame (regions are now columns)"""
    df_reduced = df_reduced.T
    
    """ Prepare colormap"""
    cm = plt.get_cmap('gist_rainbow')
    
    """ Prepare figure"""
    f = plt.figure(figsize=(12,7))

    """ Set title"""
    plt.title(title, color='black')

    """ Plot"""
    ax = df_reduced.plot(logy=True, legend=True, colormap=cm, ax=f.gca())

    """ Set legend right of plot"""
    ax.legend(bbox_to_anchor=(1, 1))

    """ Make plot smaller and shift left so that legend fits"""
    box = ax.get_position()
    ax.set_position([box.x0-0.05, box.y0, box.width * 0.8, box.height])
    
    """ Save plot"""
    plt.savefig(filename, dpi=300)
    
    """ Optionally show plot. May be annoying if multiple figures are plotted."""
    plt.show()

    
def main():
    """
    Main function. Handles everything.
    Arguments: None.
    Returns None.
    """
    
    """ Download and load data"""
    dfs = get_data()
    
    """ Preprocess data, combine rows for country provinces"""
    combine_list = ["Australia", "US", "Canada", "Mainland China", "China"]
    for key in dfs.keys():
        dfs[key] = preprocess(df=dfs[key], combine_list=combine_list)
    
    """ Compute additional variables"""
    dfs = compute_deaths_over_closed(dfs)
    dfs = compute_active_cases(dfs)
    dfs = compute_death_rate(dfs)
    dfs = compute_active_cases_reindexed(dfs)
    
    """ Set parameters for plotting"""
    titles = {"active_cases": "COVID-19 Active Cases", "active_cases_reindexed": "COVID-19 Active Cases (Days from Start of the Outbreak)", "deaths_over_closed": "COVID-19 Deaths over (deaths + recovered)", "death_rate": "COVID-19 Death rate"}
    filenames = {"active_cases": "covid19_active.png", "active_cases_reindexed": "covid19_active_ri.png", "deaths_over_closed": "covid19_death_over_closed.png", "death_rate": "covid19_death_rate.png"}
    row_inclusion_index_threasholds = {"active_cases": 770, "active_cases_reindexed": 500, "deaths_over_closed": 770, "death_rate": 770}
    row_inclusion_indices = {}
    #row_inclusion_indices.get(x) is None:
    #    row_inclusion_indices = dfs["cases"].iloc[:,-1] > x

    """ Plot"""
    for key in row_inclusion_index_threasholds.keys():
        row_inclusion_indices[key] = dfs["cases"].iloc[:,-1] > row_inclusion_index_threasholds[key]
        if key == "active_cases_reindexed":
            row_inclusion_indices[key] = dfs["cases"].iloc[:,-5] > row_inclusion_index_threasholds[key]
        plot(dfs[key], row_inclusion_indices.get(key), titles[key], filenames[key])



""" Main entry point"""

if __name__ == '__main__':
    main()



# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 23:43:55 2023

@author: Vite UH
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(a, b):
    """
    Reads and imports files from comma seperated values, to a python DataFrame

    Arguments:
    a: string, The name of the csv file which is to be read
    b: integer, indicates the number of rows on the csv file to be
    skipped

    Returns:
    data: A pandas dataframe with all values from the excel file
    transposed_data: The transposed pandas dataframe
    """
    data = pd.read_csv(a, skiprows=b)
    data = data.drop(['Country Code', 'Indicator Code'], axis=1)
    transposed_data = data.set_index(
        data['Country Name']).T.reset_index().rename(columns={'index': 'Year'})
    transposed_data = transposed_data.set_index('Year').dropna(axis=1)
    transposed_data = transposed_data.drop(['Country Name'])
    return data, transposed_data


# import data from the csv file
a = 'worldbank_data.csv'
b = 4

data, transposed_data = read_data(a, b)

# Slicing the dataframe to get data for the indicators of interest


def indicator_set(cy, al, gdp, frw):
    """
    Reads and selects precise indicators from world bank dataframe,
    to a python DataFrame

    Arguments:
    cy: Cereal Yields
    al: Agricultural Land
    gdp: GDP value in contrast to agriculture
    frw: Annual Fresh water

    Returns:
    ind: A pandas dataframe with specific indicators selected
    """
    ind = data[data['Indicator Name'].isin([cy, al, gdp, frw])]

    return ind


cy = 'Cereal yield (kg per hectare)'
al = 'Agricultural land (sq. km)'
gdp = 'Agriculture, forestry, and fishing, value added (% of GDP)'
frw = 'Annual freshwater withdrawals, total (billion cubic meters)'

ind = indicator_set(cy, al, gdp, frw)


# Slicing the dataframe to get data for the countries of interest
def country_set(countries):
    """
    Reads and selects country of interest from world bank dataframe,
    to a python DataFrame

    Arguments:
    countries: A list of countries selected from the dataframe
    Returns:
    specific_count: A pandas dataframe with specific countries selected
    """
    specific_count = ind[ind['Country Name'].isin(countries)]
    specific_count = specific_count.dropna(axis=1)
    specific_count = specific_count.reset_index(drop=True)
    return specific_count


# Selecting the countries specifically
countries = ['India', 'China', 'Brazil', 'France', 'Mexico',
             'Russian Federation', 'Canada', 'United States', 'Netherlands']

specific_count = country_set(countries)

# STATISTICS OF THE DATA
stats_desc = specific_count.groupby(["Country Name", "Indicator Name"])
print(stats_desc.describe())


def skew(dist):
    """ Calculates the centralised and normalised skewness of dist. """

    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)

    # now calculate the skewness
    value = np.sum(((dist-aver) / std)**3) / len(dist-1)

    return value


def kurtosis(dist):
    """ Calculates the centralised and normalised excess kurtosis of dist. """

    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)

    # now calculate the kurtosis
    value = np.sum(((dist-aver) / std)**4) / len(dist-1) - 3.0

    return value


def grp_countries_ind(indicator):
    """
    Selects and groups countries based on the specific indicators,
    to a python DataFrame

    Arguments:
    indicator: Choosing the indicator

    Returns:
    grp_ind_con: A pandas dataframe with specific countries selected
    """
    grp_ind_con = specific_count[specific_count["Indicator Name"] == indicator]
    grp_ind_con = grp_ind_con.set_index('Country Name', drop=True)
    grp_ind_con = grp_ind_con.transpose().drop('Indicator Name')
    grp_ind_con[countries] = grp_ind_con[countries].apply(
        pd.to_numeric, errors='coerce', axis=1)
    return grp_ind_con


# Giving each indicator a dataframe
c_yield = grp_countries_ind("Cereal yield (kg per hectare)")
agr_land = grp_countries_ind("Agricultural land (sq. km)")
agr_gdp = grp_countries_ind(
    "Agriculture, forestry, and fishing, value added (% of GDP)")
fresh_ww = grp_countries_ind(
    "Annual freshwater withdrawals, total (billion cubic meters)")

# Now we can check for the skewness and kurtosis of each indicator selected
print(skew(c_yield))
print(kurtosis(c_yield))

# Based on research, checking for correlation between indicators selected
ned_df = data[data['Country Name'].isin(['Netherlands'])]
ned_df = ned_df.drop(['Country Name'], axis=1)
ned_df = ned_df[ned_df[
    'Indicator Name'].isin([
        'Cereal yield (kg per hectare)',
        'Agricultural land (sq. km)',
        'Agriculture, forestry, and fishing, value added (% of GDP)',
        'Annual freshwater withdrawals, total (billion cubic meters)'])]
ned_df = ned_df.set_index('Indicator Name')
ned_df = ned_df.drop(ned_df.loc[:, '1990':'2019'], axis=1)
ned_df = ned_df.transpose()
ned_cor = ned_df.corr().round(2)

# Plotting the heatmap and specifying the plot parameters
plt.imshow(ned_cor, cmap='Accent_r', interpolation='none')
plt.colorbar()
plt.xticks(range(len(ned_cor)), ned_cor.columns, rotation=90)
plt.yticks(range(len(ned_cor)), ned_cor.columns)
plt.gcf().set_size_inches(8, 5)
plt.rcParams["figure.dpi"] = 300
plt.savefig('Heatmap Corr.png')
# Labelling of the little boxes and creation of a legend
labels = ned_cor.values
for y in range(labels.shape[0]):
    for x in range(labels.shape[1]):
        plt.text(x, y, '{:.2f}'.format(labels[y, x]), ha='center', va='center',
                 color='black')
plt.title('Correlation Map of Indicators for Netherlands')


# Setting Plot Style for the subsequent plots
plt.style.use('seaborn-whitegrid')

# Emphasis on Cereal Yields Produced across the years for countries selected
plt.figure(figsize=(12, 8))
c_yield.plot()
plt.title('Subtotal for Cereal Yields')
plt.xlabel('Years')
plt.ylabel('Cereal Produce')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.rcParams["figure.dpi"] = 300
plt.savefig('cereal yield.png')
plt.show()

# Agricultural, forestry and fishery (% GDP value) Overview
agr_gdp.iloc[-5::].plot(kind='bar', figsize=[10, 4])
plt.title('Agricultural GDP over the Years', fontsize=12, c='k')
plt.xlabel('Years', c='k')
plt.ylabel('GDP Percentage', c='k')
plt.rcParams["figure.dpi"] = 300
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.savefig('agr gdp.png')
plt.show()

# Agricultural Land Overview for Farming
labels = ['India', 'China', 'Brazil', 'France', 'Mexico',
          'Russian Federation', 'Canada', 'United States', 'Netherlands']
agrland_2018 = agr_land.loc['2018']
plt.figure(figsize=(12, 6))
plt.pie(agrland_2018, labels=labels, autopct='%.2f%%')
plt.rcParams["figure.dpi"] = 300
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Agricultural Lands in 2018')
plt.savefig('agrland_18.png')
plt.show()

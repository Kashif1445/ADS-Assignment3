

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def read_world_bank_records(file_path):
    """
    Read a World Bank record CSV file and return a pandas DataFrame.
    
    Parameters:
    -----------
    file_path: str
        File path of the CSV file containing the World Bank records.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the World Bank records.
    """
    df = pd.read_csv(file_path, skiprows=4)
    df_data_reverse=df.transpose()
    # Replace empty cells with zeroes
    df = df.fillna(0)

    # Keep only rows with mostly non-zero values
    df = df.loc[df.astype(bool).sum(axis=1) > len(df.columns) / 2]

    # Save the new dataset as a CSV file
    df.to_csv("climate_change.csv", index=False)

    # Load the preprocessed dataset
    df = pd.read_csv("climate_change.csv")

    return df,df_data_reverse

# Read the dataset
df,df_data_reverse = read_world_bank_records("dataset/API_19_DS2_en_csv_v2_5361599.csv")


# Group the data by indicator name
grouped = df.groupby('Indicator Name')
print(grouped)
# Loop through each group and save the data to a separate CSV file
for name, group in grouped:
    group.to_csv(f"{name}.csv", index=False)


# Load the preprocessed dataset
df = pd.read_csv("climate_change.csv")

# Group the data by indicator name
grouped = df.groupby('Indicator Name')

group_indices = list(grouped.groups.values())[:2]

# Loop through the selected groups and perform clustering and curve fitting
for idx in group_indices:
    group = df.iloc[idx]
    name = group['Indicator Name'].iloc[0]

    # Load the data
    data = pd.read_csv(f"{name}.csv")
    data1 = pd.read_csv(f"{name}.csv")
    # Select the 10 most recent years
    years = [str(i) for i in range(2012, 2022)]
    data = data[['Country Name'] + years]
    # Select the 10 most recent years
    old_years = [str(i) for i in range(2005, 2010)]
    data1 = data1[['Country Name'] + old_years]

    # Convert the data to a numpy array
    X = data.drop('Country Name', axis=1).to_numpy()
    X1 = data1.drop('Country Name', axis=1).to_numpy()


    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X1 = scaler.fit_transform(X1)

    try:
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)

        # Perform curve fitting
        def func(x, a, b, c):
            return a * np.exp(-b * (x-2012)) + c

        # Create the subplots
        fig, axs = plt.subplots(nrows=2, ncols=2,  figsize=(12, 4))
        fig.subplots_adjust(wspace=0.1, hspace=0.3)
        # Set font size for tick labels in subplots
        for ax in axs.flatten():
            ax.tick_params(axis='both', which='major', labelsize=7)
        # Plot the clustering results
        axs[0,0].scatter(X[:, 0], X[:, 1], c=labels)
        axs[0,0].set_title(f"{name} - KMeans Clustering for 2012-2022", fontdict={'fontsize': 7})

        # Plot the curve fitting results
        for i in range(X.shape[0]):
            popt, pcov = curve_fit(func, years, X[i, :], maxfev = 18000)
            axs[1,0].plot(years, func(np.array(years, dtype=int), *popt))
        axs[1,0].set_title(f"{name} - for 2012-2022Curve Fitting", fontdict={'fontsize': 7})
      # Perform K-Means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X1)

        # Perform curve fitting
        def func(x, a, b, c):
            return a * np.exp(-b * (x-2005)) + c

        # Plot the clustering results
        axs[0,1].scatter(X1[:, 0], X1[:, 1], c=labels)
        axs[0,1].set_title(f"{name} - KMeans Clusteringfor 2005-2010", fontdict={'fontsize': 7})

        # Plot the curve fitting results
        for i in range(X1.shape[0]):
            popt, pcov = curve_fit(func, old_years, X1[i, :], maxfev = 18000)
            axs[1,1].plot(old_years, func(np.array(old_years, dtype=int), *popt))
        axs[1,1].set_title(f"{name} - Curve Fitting 2005-2010", fontdict={'fontsize': 7})

        # Display the plot
        plt.show()

    except ValueError:
        print(f"Skipping {name}.csv due to ValueError: n_samples={X.shape[0]} should be >= n_clusters=3.") 

        # Display the plot
        plt.show()

""" Module errors. It provides the function err_ranges which calculates upper
and lower limits of the confidence interval. """




def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper



def exponential(x, a, b, c):
    return a * np.exp(b * (x - 2012)) + c


def logistic(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))


def quadratic(x, a, b, c):
    """
    Compute the quadratic function of the form: f(x) = a(x-c)^2 + b(x-c) + 1
    Parameters:

    x : array-like
    Array of values for which the function is computed
    a : float
    Coefficient of the quadratic term
    b : float
    Coefficient of the linear term
    c : float
    Coefficient of the quadratic term
    Returns:

    y : array-like
    The value of the function for each value in x.

    """
    return a * (x - c) ** 2 + b * (x - c) + 1


# Load the preprocessed dataset
df = pd.read_csv("climate_change.csv")

# Group the data by indicator name
grouped = df.groupby('Indicator Name')
group_indices = list(grouped.groups.values())[:2]

# Loop through the selected groups and perform clustering and curve fitting
for idx in group_indices:
    group = df.iloc[idx]
    name = group['Indicator Name'].iloc[0]
    # Load the data
    data = pd.read_csv(f"{name}.csv")
    # Select the 10 most recent years
    years = [str(i) for i in range(2012, 2022)]
    data = data[['Country Name'] + years]
    # Convert the data to a numpy array
    X = data.drop('Country Name', axis=1).to_numpy()

    # Create the subplots
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.1, hspace=0.3)
    # Set font size for tick labels in subplots
    axs.tick_params(axis='both', which='major', labelsize=7)

    # Plot the original data
    for i in range(X.shape[0]):
        axs.scatter(years, X[i, :], label='Original Data')
    
        # Perform curve fitting
        popt, pcov = curve_fit(quadratic, np.array(years, dtype=int), X[i, :], maxfev=55000)
    
        # Generate curve fit
        fit_y = quadratic(np.array(years, dtype=int), *popt)

        # Generate confidence range
        lower, upper = err_ranges(np.array(years, dtype=int), quadratic, popt, np.sqrt(np.diag(pcov)))

        # Plot curve fit and confidence range
        axs.plot(years, fit_y, label='quadratic Curve Fit')
        axs.fill_between(years, lower, upper, alpha=0.3)
    
        # Predict 10 years into the future
        future_years = [str(i) for i in range(2022, 2032)]
        future_x = np.array(future_years, dtype=int)
        future_y = quadratic(future_x, *popt)

        # Generate confidence range for future values
        future_lower, future_upper = err_ranges(future_x, quadratic, popt, np.sqrt(np.diag(pcov)))

        # Plot future predictions and confidence range
        axs.plot(future_years, future_y, label='quadratic Future Predictions')
        axs.fill_between(future_years, future_lower, future_upper, alpha=0.3)

        # Add plot title, x and y labels and legend
        axs.set_title(f"{name} - quadratic Curve Fitting", fontdict={'fontsize': 7})
        axs.set_xlabel('Year', fontsize=7)
        axs.set_ylabel('Value', fontsize=7)
      
        
        plt.show()
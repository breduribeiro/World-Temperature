import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import streamlit as st

# Fuction to data preparation (load the files and drop not used variables)


@st.cache_data
def data_preparation():
    # Importing the co2 dataset
    df_co2 = pd.read_csv("Data\owid-co2-data.csv", index_col='year')
    df_co2 = df_co2[['population', 'co2', 'primary_energy_consumption', 'methane', 'nitrous_oxide']][(
        df_co2.country == 'World') & (df_co2.index >= 1880)]

    # Importing temperature dataset
    df_temp = pd.read_csv("Data\GLB.Ts+dSST.csv", header=1, index_col="Year")
    df_temp = df_temp[df_temp.columns[:12]]

    # Replace strange values for nan
    df_temp.replace('***', np.nan, inplace=True)
    for col in df_temp.columns:
        df_temp[col] = df_temp[col].astype('float')
    df_temp['mean_temp'] = df_temp.mean(
        axis=1, skipna=True)
    df_temp = df_temp.mean_temp

    # Joing the both datasets in one
    df = df_co2.join(df_temp)
    df.index = df.index.astype(str)
    return df

# Function to calculate the temperature differences between each year and 3, 5 and 10 years before


def diff(df):
    # inserting new variables with 3, 5 and 10 year's before of world temperature
    df['mean_temp_3'] = df['mean_temp']-df['mean_temp'].shift(3)
    df['mean_temp_5'] = df['mean_temp']-df['mean_temp'].shift(5)
    df['mean_temp_10'] = df['mean_temp']-df['mean_temp'].shift(10)

    df = df.dropna(axis=0, how='any')
    return df

# Function to calculate the mean temperature in groups of 5 years


def mean(df):
    # grouping each 5 years by mean of variables
    df_mean = pd.DataFrame(
        columns=["period", "mean_temp", "mean_pop", "mean_co2", "mean_cons"])

    for year in df.index:
        if (int(year)-2) % 5 == 0:
            period = f'{int(year)}-{int(year)+4}'
            mean_temp = 0
            mean_pop = 0
            mean_co2 = 0
            mean_cons = 0
            for i in range(5):
                mean_temp += df.mean_temp.shift(-i)[df.index == year]
                mean_pop += df.population.shift(-i)[df.index == year]
                mean_co2 += df.co2.shift(-i)[df.index == year]
                mean_cons += df.primary_energy_consumption.shift(-i)[
                    df.index == year]
            df_mean = df_mean.append({'period': period,
                                      'mean_temp': mean_temp.item()/5,
                                      'mean_pop': mean_pop.item()/5,
                                      'mean_co2': mean_co2.item()/5,
                                      'mean_cons': mean_cons.item()/5}, ignore_index=True)
    df_mean = df_mean.set_index('period').dropna(axis=0, how='any')
    return df_mean

# Function to split the dataset between features dataset and target (mean_temp) dataset


def split(dataset):
    df_features = dataset.drop('mean_temp', axis=1)
    df_target = pd.DataFrame(dataset['mean_temp'])
    return(df_features, df_target)

# Function to standartize the features dataset


def scale(X_train, X_test):
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    return(X_train_scaled, X_test_scaled, sc)


# Loading the dataset
df = data_preparation()

# Initial screen of Streamlit app
st.title("WORLD TEMPERATURE")
st.image("Data\World_temp.png")
st.subheader("HOW TEMPERATURE HAS CHANGED OVER THE DECADES")
"Breno Ribeiro"
"Kaitlyn Chebowski"
"Vincenzo Mazzeo"

expand = st.expander("Introduction", expanded=False)
with expand:
    """The main goal of this project is analyse the influence of CO2 emissions and population growth over the
    global temperature's increasing.
    \nThe data is origined from two tables:
    \n[CO2 and Greenhouse Gas Emissions](https://github.com/owid/co2-data), composed of 74 variables and describes
    the year of observation, from , the country responsible for the emission, its population and gross domestic product
    and the detailed gas emission, including CO2, methane and nitrous oxide. Also details the energy consumption and the origin of the gas emission.
    \n[World Temperature Variation](https://data.giss.nasa.gov/gistemp/), composed of 19 variables and describes the year of observation
    (varying from 1880 until the present), the monthly temperature variation and the 3-months mean temperature.
    \nFor the sake of simplicity and available data, the data was reduced, filtering as world CO2 emission, population and primary energy consumption
    since 1965 (due lack o observations before that for primary energy consumption) and mean temperature variation by year.
    \nTwo diferente approaches were consider:
    \n1. Introduce the difference between the temperature variation of each year and 3, 5, and 10 years before
    \n2. Group by each 5, calculating the average of each variable along the years."""

# Presentation of the dataset
st.dataframe(df, use_container_width=True)


# Presentation of the normalized data to best visualization
sc = StandardScaler()
df_norm = pd.DataFrame(sc.fit_transform(df), columns=df.columns,
                       index=df.index)

fig = plt.figure()
plt.title("Normalized Dataset")
for col in df_norm.columns:
    df_norm[col].plot()
plt.legend()
st.pyplot(fig)

# Droping methane and nitrous_oxide from the dataset (scenario 3)
df.drop(['methane', 'nitrous_oxide'], axis=1, inplace=True)

# Radio options at lateral bar for approach (temperature difference or group mean each 5 years)
# and model to prediction (Linear Regression or Ridge Regression)
approach = st.sidebar.radio("Approach",
                            ('Temperature difference', 'Group mean 5 years'))

model_str = st.sidebar.radio("Model",
                             ("Linear Regression", "Ridge Regression"))

reduction_co2 = st.sidebar.select_slider(
    "Reduction % CO2 in 5 years", range(-15, 1, 1), 0)

# Preparation of dataset according to chosen approach
if approach == 'Temperature difference':
    df_new = diff(df)
    df_new.index = df_new.index.astype(str)

else:
    df_new = mean(df)

# Presentation of dataset with chosen approach
st.dataframe(df_new, use_container_width=True)

# Separation of dataset between features and targets
df_features, df_target = split(df_new)

# Separation of dataset between X_train, X_test, y_train and y_test
X_train, X_test, y_train, y_test = train_test_split(
    df_features, df_target, test_size=0.2, shuffle=False)

# Standardization of X_train and X_test dataset
X_train_scaled, X_test_scaled, sc = scale(X_train, X_test)

# Standardizaton of features dataset to be used at complete dataset score
df_features_scaled = pd.DataFrame(sc.transform(df_features), columns=df_features.columns,
                                  index=df_features.index.astype(str))

# Modeling according to chosen model
# Linear Regression
if model_str == "Linear Regression":
    model = LinearRegression().fit(X_train_scaled, y_train)

# or Ridge Regression
else:
    ridge_score = []
# Choosing the best parameter for alpha
    alpha = np.arange(0, 20.01, 0.01)
    for item in alpha:
        ridge = Ridge(alpha=item)
        clf_ridge = ridge.fit(X_train_scaled, y_train)
        pred_rid = clf_ridge.predict(X_test_scaled)
        score = clf_ridge.score(X_test_scaled, y_test)
        ridge_score.append(score)
    max_index = ridge_score.index(max(ridge_score))
    max_alpha = alpha[max_index]
    model = Ridge(alpha=max_alpha).fit(X_train_scaled, y_train)

# Scores for train dataset, test dataset and complet dataset
columns = st.columns(3)
X = [X_test_scaled, X_train_scaled, df_features_scaled]
y = [y_test, y_train, df_target]
dataset = ['test', 'train', 'complete']
for i in range(3):
    with columns[i]:
        pred = model.predict(X[i])
        MSE = mean_squared_error(y[i], pred)
        RMSE = math.sqrt(MSE)
        score = model.score(X[i], y[i])
        f"Score for {model_str} with {dataset[i]} dataset"
        df_score = {f'{model_str} Scores': {'score': round(score, 2),
                                            'MSE': round(MSE, 2),
                                            'RMSE': round(RMSE, 2)}}
        st.dataframe(df_score)


# Creation of dataset pro prediction
df_fut = df_new.drop("mean_temp", axis=1).reset_index()

# Considering approach 1 - difference of temperatures
if approach == "Temperature difference":
    index = ['2022', '2023', '2024', '2025', '2026']
    df_fut = pd.DataFrame(columns=['population', 'co2', 'primary_energy_consumption',
                          'mean_temp_3', 'mean_temp_5', 'mean_temp_10'], index=index)

    df_fut = pd.concat([df_new.drop("mean_temp", axis=1), df_fut])

# Linearization of features variables
    for col in df_fut.columns:
        # Considering CO2 reduction
        if col == 'co2' and reduction_co2 < 0:
            for i in range(57, 62):
                df_fut[col].iloc[i] = df_fut[col].iloc[i-1] + \
                    ((reduction_co2/100)*df_fut[col].iloc[56])/5

        else:
            xp = np.polyfit(x=df_new.index.astype(int), y=df_new[col], deg=1)
            y_fitted = np.polyval(xp, np.linspace(
                int(df_new.index[0]), int(df_new.index[-1]), 57))
            delta = y_fitted[-1]-y_fitted[-2]

            for i in range(57, 62):
                df_fut[col].iloc[i] = df_fut[col].iloc[i-1]+delta

# Ploting graphs of predicted fetures, less mean temp variation
        if not col.startswith('mean_temp'):
            fig, ax = plt.subplots()
            plt.plot(df_fut[col].iloc[:57])
            plt.plot(df_fut[col].iloc[56:], 'r--')
            plt.title(col)
            plt.setp(ax.get_xticklabels(), rotation=90)
            ax.tick_params(axis='both', labelsize=6)
            st.pyplot(fig)
# Preparation of predicted variation temperatures
    df_fut_scaled = pd.DataFrame(sc.transform(df_fut), columns=df_fut.columns,
                                 index=df_fut.index.astype(str))

    pred_fut = pd.DataFrame(model.predict(df_fut_scaled),
                            columns=['mean_temp'], index=df_fut.index.astype(str))
    mean_temp_2026 = pred_fut[-1:]
    df_diff = pd.concat([pred_fut, df_fut], axis=1)
    df_fut = pd.concat([pred_fut, df_fut], axis=1)
    df_diff = diff(df_diff)
    df_fut.iloc[-5:] = df_diff.iloc[-5:]

# Iterating the predicted variation temperature until converges to difference of temperature between 2 steps less then 0.1%
    for i in range(100):
        df_fut = df_fut.drop('mean_temp', axis=1)
        df_fut_scaled = pd.DataFrame(sc.transform(df_fut), columns=df_fut.columns,
                                     index=df_fut.index.astype(str))
        pred_fut = pd.DataFrame(model.predict(df_fut_scaled),
                                columns=['mean_temp'], index=df_fut.index.astype(str))
        error = abs((pred_fut[-1:]-mean_temp_2026)/mean_temp_2026).values
        mean_temp_2026 = pred_fut[-1:]
        df_diff = pd.concat([pred_fut, df_fut], axis=1)
        df_fut = pd.concat([pred_fut, df_fut], axis=1)
        df_diff = diff(df_diff)
        df_fut.iloc[-5:] = df_diff.iloc[-5:]
        if error <= 0.0001:
            break
    df_fut = df_fut.drop('mean_temp', axis=1)

# Ploting graphs of predicted mean temp variation
    for col in df_fut.columns:
        if col.startswith('mean_temp_'):
            fig, ax = plt.subplots()
            plt.plot(df_fut[col].iloc[:57])
            plt.plot(df_fut[col].iloc[56:], 'r--')
            plt.title(col)
            plt.setp(ax.get_xticklabels(), rotation=90)
            ax.tick_params(axis='both', labelsize=6)
            st.pyplot(fig)

# Considering approach 2 - mean temperature for every 5 years
else:
    index = ['2022-2026']
    df_fut = pd.DataFrame(
        columns=['mean_pop', 'mean_co2', 'mean_cons', ], index=index)

# Linearization of features variables
    df_fut = pd.concat([df_new.drop("mean_temp", axis=1), df_fut])
    for col in df_fut.columns:
        # Considering CO2 reduction
        if col == 'mean_co2' and reduction_co2 < 0:
            df_fut[col].iloc[11] = df_fut[col].iloc[10] + \
                ((reduction_co2/100)*df_fut[col].iloc[10])/5
        else:
            df_new = df_new.reset_index()
            xp = np.polyfit(x=df_new.index, y=df_new[col], deg=1)
            y_fitted = np.polyval(xp, np.linspace(
                int(df_new.index[0]), int(df_new.index[-1]), 11))
            delta = y_fitted[-1]-y_fitted[-2]
            df_fut[col].iloc[11] = df_fut[col].iloc[10]+delta

# Ploting predicted features
        fig, ax = plt.subplots()
        plt.plot(df_fut[col].iloc[:11])
        plt.plot(df_fut[col].iloc[10:], 'r--')
        plt.title(col)
        plt.setp(ax.get_xticklabels(), rotation=90)
        ax.tick_params(axis='both', labelsize=6)
        st.pyplot(fig)

st.dataframe(df_fut, use_container_width=True)

# Final mean temperature prediction
df_fut_scaled = pd.DataFrame(sc.transform(df_fut), columns=df_fut.columns,
                             index=df_fut.index.astype(str))

pred_fut = model.predict(df_fut_scaled)
pred_fut = np.insert(pred_fut, 0, df_target.mean_temp[-1:].values)

if approach == "Temperature difference":
    x = ['2021', '2022', '2023', '2024', '2025', '2026']
    y = pred_fut[-6:]
    xytext = (-5, -30)

else:
    x = ['2017-2021', '2022-2026']
    y = pred_fut[-2:]
    xytext = (-50, 0)


# Graph of predictions
fig1, ax1 = plt.subplots()
plt.title(f"Prediction for {model_str}")
plt.plot(pred, label='Predict Temperature')
plt.plot(df_target, label='Real Temperature')
plt.plot(x, y, 'r--', label="Predict Temperature next 5 years")
plt.legend()
plt.setp(ax1.get_xticklabels(), rotation=90)
ax1.tick_params(axis='both', labelsize=6)

plt.annotate(text=round(pred_fut[-1], 2),
             xy=(x[-1], pred_fut[-1]),
             xycoords='data',
             xytext=xytext,
             textcoords='offset points',
             arrowprops=dict(arrowstyle="->", color='black')
             )
st.pyplot(fig1)

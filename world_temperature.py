import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import streamlit as st


@st.cache_data
def data_preparation():
    # Importing the co2 dataset
    df_co2 = pd.read_csv("Data\owid-co2-data.csv", index_col='year')
    df_co2 = df_co2[['population', 'co2', 'primary_energy_consumption', 'methane', 'nitrous_oxide']][(
        df_co2.country == 'World') & (df_co2.index >= 1880)]

    # Importing temperature dataset
    df_temp = pd.read_csv("Data\GLB.Ts+dSST.csv", header=1, index_col="Year")
    df_temp = df_temp[df_temp.columns[:12]]
    df_temp['mean_temp'] = df_temp.mean(axis=1, numeric_only=True)
    df_temp = df_temp.mean_temp

    # Joing the both datasets in one
    df = df_co2.join(df_temp)
    return df


def diff(df):
    # inserting new variables with 3, 5 and 10 year's before of world temperature
    df['mean_temp_3'] = df['mean_temp']-df['mean_temp'].shift(3)
    df['mean_temp_5'] = df['mean_temp']-df['mean_temp'].shift(5)
    df['mean_temp_10'] = df['mean_temp']-df['mean_temp'].shift(10)

    df = df.dropna(axis=0, how='any')

    return df


def mean(df):
    # grouping each 5 years by mean of variables
    df_mean = pd.DataFrame(
        columns=["period", "mean_temp", "mean_pop", "mean_co2", "mean_cons"])
    for year in df.index:
        if year % 5 == 0:
            period = f'{year}-{year+4}'
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


def split(dataset):
    # Separate feature values from target values
    df_features = dataset.drop('mean_temp', axis=1)
    df_target = pd.DataFrame(dataset['mean_temp'])

    return(df_features, df_target)


def scale(X_train, X_test):
    # Standartization of the feature variables
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    return(X_train_scaled, X_test_scaled, sc)


def score_(model, X, y):
    pred = model.predict(X)
    MSE = mean_squared_error(y, pred)
    RMSE = math.sqrt(MSE)
    score = model.score(X, y)
    return(pred, MSE, RMSE, score)


df = data_preparation()


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

sc = StandardScaler()
df_norm = pd.DataFrame(sc.fit_transform(df), columns=df.columns,
                       index=df.index.astype(str))
st.dataframe(df_norm)
fig = plt.figure()
plt.title("Normalized Dataset")
for col in df_norm.columns:
    df_norm[col].plot()
plt.legend()
st.pyplot(fig)

df.drop(['methane', 'nitrous_oxide'], axis=1, inplace=True)

approach = st.sidebar.radio("Approach",
                            ('Temperature difference', 'Group mean 5 years'))

model_str = st.sidebar.radio("Model",
                             ("Linear Regression", "Ridge Regression"))

if approach == 'Temperature difference':
    df_new = diff(df)
    df_new.index = df_new.index.astype(str)

else:
    df_new = mean(df)


st.dataframe(df_new, use_container_width=True)

df_features, df_target = split(df_new)
X_train, X_test, y_train, y_test = train_test_split(
    df_features, df_target, test_size=0.2, shuffle=False)
X_train_scaled, X_test_scaled, sc = scale(X_train, X_test)

df_features_scaled = pd.DataFrame(sc.transform(df_features), columns=df_features.columns,
                                  index=df_features.index.astype(str))

if model_str == "Linear Regression":
    model = LinearRegression().fit(X_train_scaled, y_train)

else:
    model = Ridge().fit(X_train_scaled, y_train)

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

fig, ax = plt.subplots()
plt.title(f"Prediction for {model_str}")
plt.plot(pred, label='Predict Temperature')
plt.plot(df_target, label='Real Temperature')
plt.legend()
plt.setp(ax.get_xticklabels(), rotation=90)
ax.tick_params(axis='both', labelsize=6)
st.pyplot(fig)

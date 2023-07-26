import pandas as pd
import matplotlib.pyplot as plt

weather = pd.read_csv("E:/karachi data.csv", index_col="DATE")
weather.head()
null_pct = weather.apply(pd.isnull).sum()/weather.shape[0]
weather.apply(pd.isnull).sum()
valid_columns = weather.columns[null_pct < 0.5]



valid_columns


# In[8]:


weather = weather[valid_columns].copy()


# In[9]:


weather.columns


# In[10]:


weather = weather.ffill()


# In[11]:


weather.apply(pd.isnull).sum()


# In[12]:


weather


# In[13]:


weather.dtypes


# In[14]:


weather.index = pd.to_datetime(weather.index)


# In[15]:


weather.index


# In[16]:


weather.index.year


# In[17]:


weather.index.year.value_counts().sort_index()


# In[18]:


weather["PRCP"].plot()


# In[19]:


weather["target"] = weather.shift(-1)["TMAX"]


# In[20]:


weather


# In[21]:


weather = weather.ffill()


# In[22]:


weather


# In[23]:


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


# In[ ]:





# In[24]:


weather.fillna(0, inplace=True)


# In[25]:


correlation_matrix = weather.corr()


# In[26]:


print(correlation_matrix)


# In[27]:


predictors = weather.columns[~weather.columns.isin(["target", "NAME", "STATION"])]


# In[28]:


predictors


# In[29]:


weather_cleaned = weather.dropna(axis=1)


# In[30]:


print(weather_cleaned)


# In[ ]:





# In[31]:


def backtest(weather, model, predictors, target_col="temperature", start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]
        test = weather.iloc[i:(i+step), :]

        # Ensure target column is present in the DataFrame
        if target_col not in train.columns:
            raise ValueError(f"Column '{target_col}' not found in the DataFrame.")

        # Drop non-numeric columns from input features
        train_numeric = train[predictors].select_dtypes(include='number')
        test_numeric = test[predictors].select_dtypes(include='number')

        model.fit(train_numeric, train[target_col])

        preds = model.predict(test_numeric)

        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test[target_col], preds], axis=1)

        combined.columns = ["actual", "prediction"]

        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()

        all_predictions.append(combined)

    return pd.concat(all_predictions, axis=0)


# In[32]:


weather.columns = weather.columns.str.strip()


# In[33]:


print(weather.columns)


# In[34]:


print(weather.info())


# In[35]:


predictions = backtest(weather, rr, predictors, target_col="target")  # Replace "target" with the correct column name


# In[ ]:


predictions


# In[281]:


plt.plot(predictions.index, predictions['actual'], label='Actual', color='blue')

# Plot the predicted values
plt.plot(predictions.index, predictions['prediction'], label='Prediction', color='orange')

# Plot the difference values
plt.plot(predictions.index, predictions['diff'], label='Difference', color='green')

# Add labels and title to the plot
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Actual vs. Prediction vs. Difference')

# Add a legend to distinguish between lines
plt.legend()

# Display the plot
plt.show()


# In[ ]:





# In[317]:


from sklearn.metrics import mean_absolute_error

mean_absolute_error(predictions["actual"], predictions["prediction"])


# In[318]:


predictions["diff"].mean()


# In[319]:


def pct_diff(old, new):
    return (new - old)  / old

def compute_rolling(weather, horizan, col):
    label = f"rolling_{horizan}_{col}"
    
    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])
    return weather

rolling_horizons = [3, 14]

for horizon in rolling_horizons:
    for col in ["TMAX", "TMAX", "PRCP"]:
        weather = compute_rolling(weather, horizon, col)


# In[320]:


weather


# In[321]:


weather = weather.iloc[14:,:]


# In[322]:


weather


# In[323]:


weather.index = pd.to_datetime(weather.index)


# In[324]:


for col in ["TMAX", "TMIN", "PRCP"]:
    if col in weather.columns:
        weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month).apply(expand_mean)

# Calculate expanding mean values for columns 'TMAX', 'TMIN', and 'PRCP' based on day of the year
for col in ["TMAX", "TMIN", "PRCP"]:
    if col in weather.columns:
        weather[f"day_avg_{col}"] = weather[col].groupby(weather.index.day_of_year).apply(expand_mean)

# Fill any NaN values with 0
weather_filled = weather.fillna(0)


# In[325]:


weather


# In[326]:


predictors = weather.columns[~weather.columns.isin(["target", "NAME", "STATION"])]


# In[327]:


print(weather.isnull().sum())


# In[328]:


weather.dropna(inplace=True)


# In[329]:


weather.fillna(0, inplace=True)


# In[330]:


weather.fillna(weather.mean(), inplace=True)


# In[331]:


predictors


# In[335]:


predictions


# In[336]:


predictions.sort_values("diff", ascending=False)


# In[337]:


weather.loc["1990-03-07":"1990-03-17"]


# In[338]:


predictions["diff"].round().value_counts().sort_index().plot()


# In[339]:


print(weather.columns)


# In[ ]:





# In[ ]:





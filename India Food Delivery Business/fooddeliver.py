import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mysql.connector

#  Step 1 : Load the dataset
df=pd.read_csv('food_delivery_50000_rows.csv')
print(df)

# Step 2: check Data types all columns
print("Data types of all columns:")
print(df.dtypes)

# Step 3: Identity columns with non-numeric data types
non_numeric_columns= df.select_dtypes(include=['object']).columns
print("\nColumns with non-numeric data types:")
print(non_numeric_columns)

# Step 4: Handle missing values 
# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# step 5: Drop rows with missing values
print(df.dropna(inplace=True))

# step 6: Check for duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())


# Step 7: Convert date columns to datetime format
# Assuming 'Order_Timestamp' and 'Delivery_Timestamp' are the date columns

df['Order_Time'] = pd.to_datetime(df['Order_Timestamp'], format='%d-%m-%Y %H:%M')  # Parsing the date correctly
df['Order_Year'] = df['Order_Time'].dt.year  # Extracting the year from the 'Date' column
df['Order_Month'] = df['Order_Time'].dt.month  # Extracting the month for seasonal analysis
df['Order_Hour'] = df['Order_Time'].dt.hour  # Extracting the day for daily trends

df['Deliver_Time'] = pd.to_datetime(df['Delivery_Timestamp'], format='%d-%m-%Y %H:%M', errors='coerce')
df['Deliver_Year'] = df['Deliver_Time'].dt.year  # Extracting the year from the 'Date' column
df['Deliver_Month'] = df['Deliver_Time'].dt.month  # Extracting the month for seasonal analysis
df['Delivery_Hour'] = df['Deliver_Time'].dt.hour  # Extracting the hour for hourly trends

# Categorize time into segments
def time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['Time_Segment'] = df['Order_Hour'].apply(time_of_day)
print("Time segments categorization complete.")
print(df['Time_Segment'].head())

df.to_csv('food_delivery_50000_rows.csv', index=False)

# step 8: EDA

# Bar plot Order Counts bye time segment(Morining, Afternoon,Evening,Night)
"""plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Time_Segment', order=['Morning', 'Afternoon', 'Evening', 'Night'])
plt.title('Order Volume by Time Segment')
plt.xlabel('Time of Day')
plt.ylabel('Number of Orders')
plt.tight_layout()
plt.show()"""

# Bar plot Order Counts by Day of Week
"""plt.figure(figsize=(10, 5))
order_hourly = df.groupby('Order_Hour').size()
plt.plot(order_hourly.index, order_hourly.values, marker='o',mfc='red')
plt.title('Order Trends by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Orders')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()"""

# Show Correlation heatmap
"""plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()"""

# show proportion of orders by time segment
"""plt.figure(figsize=(10, 6))
segment_counts = df['Time_Segment'].value_counts()
plt.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Order Proportion by Time Segment')
plt.show()"""

# Group by Restaurant_ID to find average delivery time
avg_delivery_time = df.groupby('Restaurant')['Delivery_Duration_Minutes'].mean().reset_index()
print(avg_delivery_time.head())

# Top 5 restaurants with highest average delivery times
top_slowest = avg_delivery_time.sort_values(by='Delivery_Duration_Minutes', ascending=False).head(10)
print(top_slowest)

df.to_csv('Cleaned_Food_Deliver_Data.csv',index=False)

# Top 5 Restaurants with Highest Avg Delivery Time
"""plt.figure(figsize=(10, 6))
sns.barplot(data=top_slowest, x='Restaurant', y='Delivery_Duration_Minutes', palette='Reds_r')
plt.title('Top 5 Restaurants with Highest Avg Delivery Time')
plt.xlabel('Restaurant Name')
plt.ylabel('Delivery_Duration_Minutes')
plt.tight_layout()
plt.show()"""

# Step 9: Predictive Modeling
# Prepare the data for modeling

# Clean column names
df.columns = df.columns.str.strip()

# Clean Price column
df['Price'] = df['Price'].replace('[₹?]', '', regex=True).astype(float)
features =['Price', 'Weekend', 'Holiday', 'Delivery_Duration_Minutes']
target = 'Quantity'

# One-hot encode categorical features like Weather if needed
df_encoded = pd.get_dummies(df[['Weather']], drop_first=True)

# Combine all features
X = pd.concat([df[features], df_encoded], axis=1)
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')




















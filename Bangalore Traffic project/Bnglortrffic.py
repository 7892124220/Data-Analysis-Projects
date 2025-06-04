import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Datset
df=pd.read_csv('Bangloretraffic.csv')
print(df.head())

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

# Step 5: Fill missing Values
df['Accident_Severity'] = df['Accident_Severity'].fillna(df['Accident_Severity'].mode()[0])
print(df['Accident_Severity'])

df['Accident_Cause'] = df['Accident_Cause'].fillna(df['Accident_Cause'].mode()[0])
print(df['Accident_Cause'])

print(df.isnull().sum())

# step 6: Check for duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

df=df.drop_duplicates()
print("Number of duplicate rows after removal:", df.duplicated().sum())

# Step 7: Convert timestamp to datetime and extract hour/month
df['Time']= pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
df['hour'] = df['Time'].dt.hour
df['month'] = df['Time'].dt.month

#Step 8: Traffic Valume 
df['Traffic_volume'] = df['Cars'] + df['Buses'] + df['Trucks']
print(df['Traffic_volume'])

# Step 10: Total Vehicles
total_cars=df['Cars'].sum()
print("Total Cars: ", total_cars)
total_buses=df['Buses'].sum()
print("Total Buses: ", total_buses)
total_trucks=df['Trucks'].sum()
print("Total Trucks: ", total_trucks)

# Step 11: Total Traffic Volume
total_traffic_volume=df['Traffic_volume'].sum()
print("Total Traffic Volume: ", total_traffic_volume)
# Check unique values in the severity column 
print(df['Accident_Severity'].unique())
# Check unique values in the Accident Cause column 
print(df['Accident_Cause'].unique())

df.to_csv("Bangloretraffic.csv",index=False)

# MYSQL CONNECTION TO PYTHON
"""mydb=mysql.connector.connect(
  host="localhost",
  user='root',
  password='system',
  database='traffic202501'
)

query="select * from traficrec"
print(query)
df = pd.read_sql(query, mydb)
print(df)
mydb.close()"""


# EDA: Peak vs. off-peak traffic hours
"""hourly_traffic = df.groupby('hour')['Traffic_volume'].mean()
plt.figure(figsize=(10,5))
sns.lineplot(x=hourly_traffic.index, y=hourly_traffic.values,marker='o', color='navy',mfc='red')
plt.title('Average Traffic Volume by Hour',color='darkred')
plt.xlabel('Hour')
plt.ylabel('Avg Traffic Volume')
plt.tight_layout()
plt.show()"""

# Most congested areas
"""location_traffic = df.groupby('Location')['Traffic_volume'].mean().sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=location_traffic.values, y=location_traffic.index, palette='Reds')
plt.title('Most Congested Areas (Avg Traffic Volume)')
plt.xlabel('Avg Traffic Volume')
plt.ylabel('Location')
plt.tight_layout()
plt.show()"""

# Impact of weather on traffic slowdown
"""plt.figure(figsize=(8,5))
sns.boxplot(x='Weather', y='Traffic_volume', data=df)
plt.title('Traffic Volume by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Traffic Volume')
plt.tight_layout()
plt.show()"""


# Distribution of Accident Causes
"""cause_counts = df['Accident_Cause'].value_counts().reset_index()
cause_counts.columns = ['Accident_Cause', 'count']
plt.figure(figsize=(10, 6))
sns.barplot(data=cause_counts, x='Accident_Cause', y='count', palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.title("Distribution of Accident Causes")
plt.xlabel("Accident Cause")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()"""

# Distribution of Weather Conditions During Accidents
"""weather_counts = df['Weather'].value_counts().reset_index()
weather_counts.columns = ['Weather', 'count']
plt.figure(figsize=(8, 5))
sns.barplot(data=weather_counts, x='Weather', y='count', palette='coolwarm')
plt.xticks(rotation=30, ha='right')
plt.title("Distribution of Weather Conditions During Accidents")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()"""

# Correlation heatmap
"""plt.figure(figsize=(10,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()"""

    

# LINEAR REGRESSION :

# Clean column names
df.columns = df.columns.str.strip()

# Encode categorical features
df['Weather'] = df['Weather'].map({'Clear':0, 'Rain':1, 'Fog':2,'Cloudy':3})
print(df['Weather'])


# One-hot encode categorical features like Weather if needed
#df_encoded = pd.get_dummies(df[['Weather']], drop_first=True)

df['Weather_Code'] = df['Weather'].astype('category').cat.codes

df['Location_Code'] = df['Location'].astype('category').cat.codes

# Combine all features
X = df[['hour','month','Weather_Code','Location_Code']]
y = df['Traffic_volume'] 
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
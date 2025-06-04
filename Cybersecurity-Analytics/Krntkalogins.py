import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Datset
df=pd.read_csv('CyberKarnatakalogins.csv')
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

# Step 7: Fill missing values
df['Access_Duration'] = df['Access_Duration'].fillna(0)
print(df['Access_Duration'])

print(df.isnull().sum())

#Step 7: Convert timestamp to datetime and extract hour/month
df['Time'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M') 
print(df['Time'])
df['hour'] = df['Time'].dt.hour
print(df['hour'])
df['month'] = df['Time'].dt.month
print(df['month'])

df.to_csv("Karnatakalogins.csv",index=False)

# 1. Identify users logging in from unusual locations
user_location_mode = df.groupby('User_ID')['Location'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
print("Most common location for each user:")
print(user_location_mode)

df['usual_location'] = df['User_ID'].map(user_location_mode)
print("Usual location for each user:")
print(df['usual_location'])


# Flag logins from locations different than the user's usual location
df['unusual_location'] = df['Location'] != df['usual_location']
print("Flagging unusual locations:")
print(df['unusual_location'])

unusual_logins = df[df['unusual_location']]
print("Users logging in from unusual locations:")
print(unusual_logins[['User_ID', 'Location', 'usual_location','hour','month']])


# 2. Detect login attempts at odd hours (e.g., between 0:00-5:00)
df['odd_hour'] = df['hour'].apply(lambda x: 1 if (x >= 0 and x <= 5) else 0)
print("Flagging odd hour :")
print(df['odd_hour'].head())

odd_hour_logins = df[df['odd_hour'] == 1]
print("Login attempts at odd hours (00:00-05:00):")
print(odd_hour_logins[['User_ID', 'hour', 'Timestamp', 'Location']])

print(df.isnull().sum())

df.to_csv('Karnatakalogins.csv',index=False)
# Bar plot: Number of unusual logins per user
"""plt.figure(figsize=(10,5))
unusual_counts = unusual_logins['User_ID'].value_counts()
sns.barplot(x=unusual_counts.index,y=unusual_counts.values,hue=unusual_counts.index,palette='viridis',legend=False)
#sns.barplot(x=unusual_counts.index, y=unusual_counts.values, palette='viridis')
plt.title('Number of Unusual Location Logins per User')
plt.xlabel('User_ID')
plt.ylabel('Unusual Login Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""


# Histogram: Distribution of odd-hour logins
"""plt.figure(figsize=(8,5))
sns.countplot(x='hour', data=odd_hour_logins, palette='magma')
plt.title('Login Attempts at Odd Hours (00:00-05:00)')
plt.xlabel('Hour of Login')
plt.ylabel('Number of Logins')
plt.tight_layout()
plt.show()"""

# Pie chart of normal vs. unusual logins
"""plt.figure(figsize=(6,6))
labels = ['Usual Location', 'Unusual Location']
sizes = [len(df) - len(unusual_logins), len(unusual_logins)]
plt.pie(sizes, labels=labels, autopct='%1.9f%%', colors=['#66b3ff','#ff9999'])
plt.title('Proportion of Usual vs. Unusual Location Logins')
plt.show()"""


# Predictive Analysis Using Machine Learning
# Apply Linear Regression to predict suspicious (unusual) logins

# Prepare features and target
# We'll use hour, odd_hour, and (if available) Login_Status as features
features = ['hour', 'odd_hour']
if 'Login_Status' in df.columns:
    # Encode Login_Status (e.g., Success=0, Failure=1)
    df['Login_Status_Code'] = df['Login_Status'].astype('category').cat.codes
    features.append('Login_Status_Code')

X = df[features]
y = df['unusual_location'].astype(int)  

# Split data
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


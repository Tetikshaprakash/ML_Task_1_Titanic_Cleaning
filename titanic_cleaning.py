import pandas as pd
import numpy as np
df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

df.drop('Cabin', axis=1, inplace=True)

# Safer, pandas-friendly version
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

df.to_csv("Titanic-Cleaned.csv", index=False)
print("Cleaned dataset saved successfully!")


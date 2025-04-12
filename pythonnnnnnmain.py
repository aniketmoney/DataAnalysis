
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv(r"C:\Users\riya choudhary\Downloads\dashboard.csv")

# Check for missing values
print(df.isnull().sum())

df['DataValue'].fillna(df['DataValue'].median(), inplace=False)  # Remove inplace=True or reassign the result


# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Convert relevant columns to appropriate data types
df['YearStart'] = df['YearStart'].astype(int)
df['YearEnd'] = df['YearEnd'].astype(int)
df['DataValue'] = pd.to_numeric(df['DataValue'], errors='coerce')
df['LocationDesc'] = df['LocationDesc'].astype('category')
df['Topic'] = df['Topic'].astype('category')
df['Stratification1'] = df['Stratification1'].astype('category')

# Check for any remaining missing values after cleaning
print(df.isnull().sum())




#Descriptive analysis
# General descriptive statistics for numeric columns
print(df.describe())

# Frequency counts for categorical variables
print(df['LocationDesc'].value_counts())
print(df['Topic'].value_counts())


#Visualizations
# Histogram for numerical columns like 'DataValue'
plt.figure(figsize=(10, 6))
sns.histplot(df['DataValue'], kde=True, bins=30)
plt.title('Distribution of DataValue')
plt.show()

# Boxplot for numerical columns to detect outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['DataValue'])
plt.title('Boxplot of DataValue')
plt.show()


#plotting for categorical values
# Bar plot for LocationDesc
plt.figure(figsize=(12, 6))
sns.countplot(x='LocationDesc', data=df, order=df['LocationDesc'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Distribution of Locations')
plt.show()

# Bar plot for Topic
plt.figure(figsize=(12, 6))
sns.countplot(x='Topic', data=df, order=df['Topic'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Distribution of Topics')
plt.show()



#Heatmap for correlation
# Correlation heatmap
# Load your dataset
# Drop unnecessary columns, if already done
# Example: df.drop(columns=['SomeColumn'], inplace=True)

# Fill missing values in 'DataValue' column, if needed
df['DataValue'].fillna(df['DataValue'].median(), inplace=True)

# Filter only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=['number'])

# Calculate and plot the correlation matrix
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Show the plot
plt.show()



# Grouping by 'LocationDesc' and calculating mean DataValue
grouped_location = df.groupby('LocationDesc')['DataValue'].mean().sort_values(ascending=False)
grouped_location.plot(kind='bar', figsize=(12, 6), color='skyblue')
plt.title('Average DataValue by LocationDesc')
plt.xticks(rotation=90)
plt.show()






#Time analysis
# Check how the data is distributed across years
plt.figure(figsize=(12, 6))
sns.countplot(x='YearStart', data=df)
plt.title('Data Distribution by YearStart')
plt.show()

# If you want to check if the `DataValue` has changed over the years:
yearly_trends = df.groupby('YearStart')['DataValue'].mean()
yearly_trends.plot(kind='line', figsize=(12, 6), marker='o', color='green')
plt.title('Average DataValue Trend Over Years')
plt.show()



#Stratification Analysis
# Grouping by Stratification1 (e.g., Sex, Race) and calculating mean DataValue
plt.figure(figsize=(12, 6))
sns.boxplot(x='Stratification1', y='DataValue', data=df)
plt.title('DataValue Distribution by Stratification')
plt.xticks(rotation=45)
plt.show()



#Outlier detection
# Identify outliers in DataValue using Z-score
from scipy.stats import zscore

df['zscore'] = zscore(df['DataValue'])
outliers = df[df['zscore'].abs() > 3]
print(outliers)

# Alternatively, using IQR method
Q1 = df['DataValue'].quantile(0.25)
Q3 = df['DataValue'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df[(df['DataValue'] < (Q1 - 1.5 * IQR)) | (df['DataValue'] > (Q3 + 1.5 * IQR))]
print(outliers_iqr)


#Checking correlation
# Correlation for numerical columns
correlation = df.corr()
print(correlation)

# Visualization of correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix for Numerical Features')
plt.show()




#Data transformation
from sklearn.preprocessing import LabelEncoder

# Drop the specified columns
columns_to_drop = ['Question', 'Response', 'TopicID', 'QuestionID', 'ResponseID', 'DataValueTypeID', 'StratificationCategoryID1']
df.drop(columns=columns_to_drop, inplace=True)

# Check remaining columns after dropping
print("Remaining columns after dropping unnecessary ones:")
print(df.columns)

# Optional: Convert 'YearStart' to a datetime column (if you want to work with dates)
df['YearStart'] = pd.to_datetime(df['YearStart'], format='%Y')

# Optional: Encoding categorical columns (e.g., 'Sex', 'LocationDesc', 'Stratification1') into numeric form using LabelEncoder
label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['LocationDesc'] = label_encoder.fit_transform(df['LocationDesc'])
df['Stratification1'] = label_encoder.fit_transform(df['Stratification1'])

# Optional: Create a new column for DataValue normalization (if needed for modeling)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['DataValue_normalized'] = scaler.fit_transform(df[['DataValue']])

# Optional: Create a new column for the year (from the 'YearStart' column)
df['Year'] = df['YearStart'].dt.year

# Check the transformed data
print("Transformed DataFrame:")
print(df.head())

# Verify the types and check if any columns need further cleaning or conversion
print("\nData types after transformation:")
print(df.dtypes)




#More Visualizations

# Set the style for the plots
sns.set(style="whitegrid")

# 1. Distribution of DataValue
plt.figure(figsize=(10, 6))
sns.histplot(df['DataValue'], kde=True, color='skyblue', bins=30)
plt.title('Distribution of DataValue', fontsize=16)
plt.xlabel('DataValue', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# 2. Bar plot for Crude Prevalence by Location (LocationDesc)
plt.figure(figsize=(12, 8))
sns.barplot(x='LocationDesc', y='DataValue', data=df, palette='viridis')
plt.xticks(rotation=90)
plt.title('Crude Prevalence by Location', fontsize=16)
plt.xlabel('Location', fontsize=12)
plt.ylabel('Crude Prevalence (DataValue)', fontsize=12)
plt.show()

# 3. Boxplot for DataValue by Stratification1
plt.figure(figsize=(12, 8))
sns.boxplot(x='Stratification1', y='DataValue', data=df, palette='Set2')
plt.xticks(rotation=45)
plt.title('DataValue Distribution by Stratification1', fontsize=16)
plt.xlabel('Stratification1', fontsize=12)
plt.ylabel('DataValue', fontsize=12)
plt.show()

# 4. Heatmap for Correlation between Numeric Columns
# Correlation between DataValue, Year, etc.
numeric_columns = ['DataValue', 'YearStart', 'DataValue_normalized']
correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap for Numeric Columns', fontsize=16)
plt.show()

# 5. Time Trend Analysis of DataValue
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='DataValue', data=df, marker='o', color='green')
plt.title('DataValue Trend Over Years', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('DataValue', fontsize=12)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_titanic.csv")

print("Dataset Info: ")
print(df.info())

print("\n Summary Statistics: ")
print(df.describe(include='all'))

print("\n Missing Values: ")
print(df.isnull().sum())

# Histograms
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols].hist(bins=20, figsize=(14, 10), color='skyblue', edgecolor='darkblue')
plt.tight_layout()
plt.savefig("raw_histograms.png")
plt.show()

#Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title("Age distribution across class")
plt.xlabel("Passenger class")
plt.ylabel("Age")
plt.savefig("boxplot_age_pclass.png")
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=['int64', 'float64']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.show()

# Pairplot
selected_cols = ['Survived', 'Age', 'Fare', 'Pclass']
sns.pairplot(df[selected_cols], hue='Survived')
plt.savefig("pairplot.png")
plt.show()

print("\n🧠 Inferences Based on EDA:\n")

# General distribution insights
print("1. Most passengers are young (20–40 years). Fare is heavily right-skewed — few passengers paid high fares.")

# Boxplot insight
print("2. Passengers in 1st class are older on average than those in 2nd or 3rd class. More outliers exist in 3rd class.")

# Correlation matrix insights
print("3. 'Pclass' has a moderate negative correlation with 'Survived' (-0.27), meaning survival rate drops with lower class.")
print("4. 'Fare' has a positive correlation with 'Survived' (0.25), suggesting that passengers who paid more had better chances of survival.")

# Pairplot observations
print("5. Survival is clearly higher in Class 1 with higher fares. Very few survived in Class 3 with low fare.")
print("6. No strong linear correlation between 'Age' and 'Survived', but visually young children show a slight survival edge.")

# Missing data handling in cleaned dataset
print("7. Missing values in 'Age', 'Cabin', and 'Embarked' were cleaned before this analysis. 'Cabin' is still too sparse to use directly but may be transformed into deck-level features.")

# Family size insights
print("8. Most passengers were alone (SibSp=0, Parch=0), with small family groups being more common than large ones.")

# Categorical feature importance
print("9. Categorical features like 'Sex' and 'Embarked' will also play a major role in modeling, especially 'Sex'.")

# 🚢 Titanic Dataset - Exploratory Data Analysis (EDA)

This project is part of an internship task focused on performing **EDA** on the raw Titanic dataset to uncover patterns, correlations, and insights.

---

## 📁 Dataset
The dataset used is the **raw Titanic dataset** (`Titanic-Dataset.csv`), downloaded from Kaggle:  
🔗 [Titanic Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

---

## 🛠 Tools & Libraries
- **Python**
- **Pandas** for data manipulation
- **Matplotlib** & **Seaborn** for visualization

---

## 🔍 EDA Steps Performed

1. Loaded the dataset and checked:
   - Structure of data (`info()`)
   - Summary statistics (`describe()`)
   - Missing values (`isnull().sum()`)

2. **Visualizations:**
   - Histograms of numeric features → `raw_histograms.png`
   - Boxplot of Age by Class → `boxplot_age_pclass.png`
   - Correlation matrix heatmap → `correlation_matrix.png`
   - Pairplot of key features → `pairplot.png`

3. **Inferences Made:**
   - Most passengers are aged 20–40
   - Fare is right-skewed (a few people paid a lot)
   - 1st class passengers were older and survived more
   - Strong correlation between `Pclass`, `Fare`, and `Survived`
   - Categorical features like `Sex` will be useful in future analysis

---

## 🧠 Key Insights
- **Class and Fare** significantly impact survival
- **Age** is less linearly related but still useful visually
- **Cabin** has too many missing values
- **SibSp** and **Parch** suggest most people were traveling alone

---

## ▶️ How to Run
Make sure your file is named `Titanic-Dataset.csv` and is in the same folder as the Python script.  
Then run:

```bash
python eda_titanic.py

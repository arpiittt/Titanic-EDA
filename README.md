# 🚢 Titanic Dataset - Exploratory Data Analysis 

This project demonstrates **Exploratory Data Analysis (EDA)** on the **cleaned Titanic dataset** to uncover hidden trends, correlations, and patterns useful for further modeling.

---

## 📁 Dataset
The dataset used is a **cleaned version** of the Titanic dataset, where missing values were handled and features were preprocessed.

Original dataset from Kaggle:  
🔗 [Titanic Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

---

## 🛠 Libraries Used
- **Python 3**
- **Pandas** – data manipulation
- **Matplotlib & Seaborn** – data visualization

---

## 📊 EDA Performed
1. **Loaded dataset** and checked:
   - Data structure (`info()`)
   - Summary statistics (`describe()`)
   - Missing values (`isnull().sum()`)

2. **Visualized:**
   - Histograms for feature distributions → `raw_histograms.png`
   - Boxplot (Age vs Class) → `boxplot_age_pclass.png`
   - Correlation matrix → `correlation_matrix.png`
   - Pairplot (with `Survived` hue) → `pairplot.png`

---

## 🧠 Key Insights

- **Age**: Most passengers fall in the 20–40 age range.
- **Fare**: Distribution is right-skewed; few passengers paid very high prices.
- **Class vs Age**: 1st class passengers are older on average than 2nd or 3rd class.
- **Survival vs Fare/Pclass**: Higher fare and lower Pclass (1st class) are associated with higher survival chances.
- **Family Features**: Most people traveled alone (SibSp and Parch = 0).
- **Feature Correlation**:
  - Pclass and Fare are strongly negatively correlated.
  - Fare and Survived are moderately positively correlated.
  - Age and Survived show a very weak relationship.

---

## 📂 How to Run

1. Make sure the cleaned dataset (`Titanic-Dataset.csv`) is in the same directory.
2. Run the script:

```bash
python eda_titanic.py

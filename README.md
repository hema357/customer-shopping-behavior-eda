# Customer Shopping Behavior & Subscription Predictive Analysis

## Project Overview
In this project, I performed an end-to-end data science analysis on a customer shopping behavior dataset. My primary goals were to explore key purchasing demographics, discover behavioral shopping patterns across categories, and build a predictive classification model to identify whether a customer will subscribe to the shopping service (subscription status analysis).

Through feature engineering and addressing class imbalance, I developed a Random Forest Classifier that models customer loyalty and subscription behavior with high accuracy.

---

## Dataset Description
The dataset contains **3,900 rows and 18 attributes** documenting customer transactions and demographic variables. Key attributes include:
* **Demographics:** Customer ID, Age, Gender, Location
* **Transaction Details:** Item Purchased, Category, Purchase Amount (USD), Size, Color, Season, Shipping Type, Payment Method
* **Loyalty & Behavior:** Review Rating, Subscription Status, Discount Applied, Promo Code Used, Previous Purchases, Frequency of Purchases

---

## Key Findings from Exploratory Data Analysis (EDA)

### 1. Demographics & Distributions
* **Age:** Customer ages are uniformly distributed between 18 and 70 years old, with an average age of approximately 44.
* **Gender:** The dataset contains a higher proportion of Male buyers (2,652) compared to Female buyers (1,248).
* **Categories:** *Clothing* is the most popular purchase category (1,737), followed by *Accessories* (1,240), *Footwear* (599), and *Outerwear* (324).

### 2. Purchase Behavior & Ratings
* **Purchase Amount:** Most items cost between \$20 and \$100, averaging roughly \$59.76.
* **Ratings:** Customer satisfaction is highly varied, with review ratings evenly spread between 2.5 and 5.0, averaging 3.75.
* **Loyalty:** On average, customers have made 25.35 previous purchases, indicating a highly recurring customer base.

### 3. Subscription & Promo Interactions
* Out of 3,900 records, **1,053 customers are active subscribers**, while **2,847 are non-subscribers**.
* **Promo Code Usage** and **Discounts Applied** strongly align with subscription behaviors.

---

## Predictive Modeling & Performance

To predict customer subscription status, I built a machine learning pipeline using a **Random Forest Classifier** with balanced class weights to address target category imbalance.

### Data Preprocessing & Feature Engineering:
1. **Categorical Encoding:** Converted categorical features (Gender, Category, Item Purchased, Location, etc.) into numerical values using One-Hot Encoding, resulting in **129 final features**.
2. **Target Encoding:** Encoded subscription status ('Yes'/'No') using a `LabelEncoder`.
3. **Validation Strategy:** Split the dataset into **70% Training** (2,730 samples) and **30% Testing** (1,170 samples) sets, utilizing stratified sampling to preserve target class balance.

### Model Results:
My model achieved an overall **Test Accuracy of 83.85%**.

#### Classification Report:
```text
              precision    recall  f1-score   support

          No       0.95      0.82      0.88       854
         Yes       0.65      0.89      0.75       316

    accuracy                           0.84      1170
   macro avg       0.80      0.86      0.81      1170
weighted avg       0.87      0.84      0.85      1170
```

#### Insights from Confusion Matrix:
* **True Negatives (Correctly predicted Non-subscribers):** 699
* **True Positives (Correctly predicted Subscribers):** 282
* **False Positives (Predicted 'Yes' but actually 'No'):** 155
* **False Negatives (Predicted 'No' but actually 'Yes'):** 34
* The model is particularly strong at capturing potential subscribers (**Recall of 89%** for the 'Yes' class), which is crucial for subscription retention campaigns.

### Top Predictors:
Through feature importance analysis, the primary factors determining whether a customer is subscribed are:
1. **Promo Code Used** (Importance: 27.55%)
2. **Discount Applied** (Importance: 26.04%)
3. **Gender** (Importance: 6.31%)

---

## Project Structure
```text
├── data/
│   └── shopping_behavior_updated.csv    # Shopping behavior dataset
├── shopping_behavior_analysis.ipynb     # Jupyter Notebook containing code and plots
└── README.md                            # Project documentation
```

---

## How to Run the Project

### 1. Prerequisites
Ensure you have Python 3.8+ and the required packages installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Execution
Run the Jupyter notebook to see the step-by-step cleaning, visualizations, and model evaluation:
```bash
jupyter notebook shopping_behavior_analysis.ipynb
```

---

## Technologies Used
* **Language:** Python
* **Data Wrangling:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Random Forest, Preprocessing, Metrics)

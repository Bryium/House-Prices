# 🏠 House Price Prediction

> Predicting property prices across India using machine learning — XGBoost-powered with **R² = 0.84**

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellowgreen?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

##  Overview

This project builds a machine learning pipeline to predict **house prices (₹ per sq ft)** across India using real-world property listings. It compares 5 regression models and selects the best-performing one — **XGBoost** — which explains **84% of price variance** on unseen data.

Key challenges tackled:
- High-cardinality categorical features (location, society) encoded with **target encoding**
- Text signals from property titles and descriptions extracted as features
- Skewed distributions corrected, interaction and polynomial features engineered

---

##  Model Performance

| Model | RMSE | R² |
|---|---|---|
| 🥇 **XGBoost** | **0.2114** | **0.8435** |
| 🥈 Random Forest | 0.2236 | 0.8250 |
| 🥉 Gradient Boosting | 0.2356 | 0.8056 |
| Ridge Regression | 0.3298 | 0.6193 |
| Lasso Regression | 0.3357 | 0.6055 |

---

##  Dataset

| Property | Detail |
|---|---|
| Source | Indian property listings |
| Size | ~170,000 rows, 19 columns |
| Target | Price in ₹ (rupees) |
| Key Features | Carpet Area, Super Area, Location, Society, Furnishing, Status, Bathrooms, Balconies |

---

##  Features & Methodology

- **5 models benchmarked:** Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost
- **High-cardinality encoding:** Target encoding for `location` and `society`
- **NLP features:** Text signals extracted from property `title` and `description`
- **Feature engineering:** Interaction terms, polynomial features, skew correction via log transforms
- **Scikit-learn pipelines** for clean, reproducible preprocessing

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` & `numpy` | Data wrangling |
| `scikit-learn` | Preprocessing, pipelines, baseline models |
| `XGBoost` | Best-performing model |
| `matplotlib` & `seaborn` | EDA & visualizations |

---

##  Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/Bryium/House-Price-Prediction.git
cd House-Price-Prediction
pip install -r requirements.txt
```

### Running the Notebook

1. Place `house_prices.csv` in the project root folder.
2. Open `House_Price_Predictor.ipynb` in Jupyter or Google Colab.
3. Run all cells top to bottom.

---

## 🔮 Making Predictions

Use the `predict_house_price()` function at the end of the notebook:

```python
predict_house_price(
    carpet_area=800,
    bathroom=2,
    location='mumbai',
    furnishing='Semi-Furnished',
    status='Ready to Move'
)
```

---

##  Project Structure

```
House-Price-Prediction/
├── House_Price_Predictor.ipynb   # Main notebook
├── requirements.txt              # Dependencies
├── .gitignore
└── README.md
```

---

##  Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---



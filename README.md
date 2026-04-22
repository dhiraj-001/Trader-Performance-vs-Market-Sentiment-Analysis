# 📊 Trader Performance vs Market Sentiment Analysis

## 📌 Overview

This project analyzes how **Bitcoin Market Sentiment (Fear & Greed Index)** impacts **trader performance** on Hyperliquid.
[Report](https://docs.google.com/document/d/1LSvOj8BSEvEyoy5WVzcRGiR8Z0cd86xIdBW0YFpH2tU/edit?usp=sharing)

The goal is to uncover:

* How different sentiment regimes affect profitability
* Whether traders perform better during fear or greed
* Actionable insights for sentiment-based trading strategies

---

## 🗂️ Project Structure

```
PrimeTrade/
│
├── dataset/
│   ├── historical_data.csv        # Trading data
│   └── fear_greed_index.csv       # Sentiment data
│
├── outputs/
│   ├── charts/                   # Generated visualizations
│   ├── tables/                   # CSV summary tables
│   └── report.md                 # Auto-generated report
│
├── analysis.py                   # Main analysis script
├── notebook.ipynb               # Optional exploration
├── requirements.txt             # Dependencies
├── report.md                    # Final report (manual/edited)
└── .gitignore
```

---

## ⚙️ Features

* 📈 Sentiment-based performance analysis
* 🔗 Data merging (trades + sentiment)
* 📊 15+ visualizations (heatmaps, bar charts, distributions)
* 📉 Statistical testing (ANOVA, Kruskal-Wallis, Spearman)
* 🧠 Insight generation + strategy recommendations
* 📄 Auto-generated report

---

## 🧪 Methodology

1. **Data Cleaning**

   * Standardized column names
   * Converted timestamps to datetime
   * Removed invalid trades

2. **Data Merging**

   * Joined trades with sentiment using date

3. **Feature Engineering**

   * Profit/Loss indicators
   * Trade size normalization
   * PnL-to-size ratio
   * Time-based features (hour, weekday)

4. **Analysis**

   * Grouped by sentiment regimes
   * Computed metrics:

     * Total PnL
     * Average PnL
     * Win rate
     * Trade count
     * Efficiency

5. **Visualization**

   * Correlation heatmap
   * PnL distribution
   * Hourly patterns
   * Sentiment vs performance

---

## 📊 Key Outputs

### 📈 Charts

* Total PnL by sentiment
* Average PnL per trade
* Win rate analysis
* Trade count distribution
* PnL distribution (boxplot)
* Hourly and weekday heatmaps
* Coin-wise performance
* Correlation heatmap

### 📋 Tables

* Summary by sentiment
* Side performance (buy/sell)
* Coin performance
* Hourly patterns
* Statistical test results

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run analysis

```bash
python analysis.py
```

---

## 📁 Output Location

All results will be saved in:

```
outputs/
├── charts/
├── tables/
└── report.md
```

---

## 📊 Sample Insights

* Extreme Greed shows highest **average PnL**
* Fear regimes have **higher trade volume**
* Win rate remains relatively stable across sentiments
* Certain coins perform better in specific sentiment regimes

---

## 🧠 Conclusion

Market sentiment plays a significant role in trading outcomes.
Instead of using a fixed strategy, traders can:

* Adjust position sizing based on sentiment
* Identify optimal trading conditions
* Improve risk management

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy

---

## 📌 Author

**Dhiraj Gogoi**
B.Tech CSE | Data Analysis & ML Enthusiast

---

## 📎 Note

This project is part of a data analysis assignment focused on:

> *Trader Performance vs Bitcoin Market Sentiment*

---

## ⭐ Future Improvements

* Add machine learning model for prediction
* Real-time sentiment integration
* Strategy backtesting

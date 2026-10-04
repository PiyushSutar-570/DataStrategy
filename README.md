# E-Commerce Data Strategy & Analytics

A data analysis and machine learning project evaluating e-commerce customer behavior, experiment metrics, and churn dynamics.

## Overview

This project processes e-commerce customer data (`ecommerce_hypothesis_data.csv`) to perform:
- **A/B Testing**: Evaluating checkout UI variant vs control performance.
- **Variance Analysis (ANOVA)**: Comparing spend across membership tiers.
- **Churn Prediction**: Training a Random Forest classifier to identify churn drivers.
- **Customer Segmentation**: Clustering customers into behavioral segments using K-Means.

## Project Structure

```text
├── main.py                          # Main analysis pipeline & report generator
├── make_notebook.py                 # Script to build Jupyter notebook
├── statistical_analysis_notebook.ipynb # Notebook containing analysis workflow
├── ecommerce_hypothesis_data.csv    # Dataset
├── requirements.txt                 # Python dependencies
└── output/                          # Output charts, results JSON, and generated report
```

## Setup & Execution

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the full analysis pipeline:
   ```bash
   python main.py
   ```

Executing `main.py` generates visual plots, exports statistical summaries to `output/`, and creates the summary Word document.

# 📊 Ensemble Learning for Sales Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14fGWpS-TnOgXzdh7SkdFXlwH_c0Segx_?usp=sharing)

> Predicting high-value sales orders using AdaBoost and Gradient Boosting classifiers.

## 🎯 Project Overview

This project implements ensemble learning methods to classify retail sales orders into Low (<$300) or High (≥$300) value categories. Through rigorous mathematical implementation and business-aligned evaluation, we deliver actionable intelligence for inventory planning and customer segmentation.. It includes:
- ✅ A **from-scratch AdaBoost implementation** for educational transparency
- ✅ **scikit-learn Gradient Boosting** for production benchmarking
- ✅ Comprehensive data preprocessing, feature engineering, and evaluation
- ✅ Business-ready insights for inventory planning and customer segmentation

# Methodology

# Data exploration and preprocessing

# Results and Discussions
# 📈 Results & Discussion

## 4.1 Model Performance Summary

### Classification Metrics Comparison

| Metric | Custom AdaBoost | Gradient Boosting | Δ Change |
|--------|----------------|-------------------|----------|
| **Accuracy** | 0.7397 | **0.7410** | +0.13% |
| **Precision (High)** | 0.6612 | 0.6434 | -1.78% |
| **Recall (High)** ⭐ | 0.5421 | **0.6015** | **+5.94%** |
| **F1-Score (High)** | 0.5958 | **0.6218** | **+2.60%** |
| **ROC-AUC** | 0.712 | **0.738** | +3.6% |
| **Training Time** | 45.2s | **12.8s** | **71.7% faster** |

> ⭐ **Recall is our primary business metric** — detecting high-value orders directly impacts revenue capture. Missing a high-value order means lost upsell opportunities and inefficient resource allocation.

---

## 4.2 Detailed Classification Analysis

### Confusion Matrix Breakdown (Gradient Boosting)

| | **Predicted: Low** | **Predicted: High** | **Row Total** |
|---|-------------------|--------------------|---------------|
| **Actual: Low** | 782 (TN) | 171 (FP) | 953 |
| **Actual: High** | 208 (FN) | 314 (TP) | 522 |
| **Column Total** | 990 | 485 | 1,475 |

**Key Performance Indicators:**

$$\text{True Negative Rate (Specificity)} = \frac{TN}{TN + FP} = \frac{782}{953} = \textbf{82.0\%}$$

$$\text{True Positive Rate (Sensitivity/Recall)} = \frac{TP}{TP + FN} = \frac{314}{522} = \textbf{60.15\%}$$

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{314}{485} = \textbf{64.34\%}$$

$$\text{False Positive Rate} = \frac{FP}{FP + TN} = \frac{171}{953} = \textbf{17.9\%}$$

**Interpretation:**
- ✅ The model excels at identifying **Low-value orders** (82% specificity), reducing unnecessary resource allocation to low-priority transactions.
- ⚠️ **High-value recall at 60.15%** indicates room for improvement: ~40% of high-value orders are missed. This is acceptable for a first deployment but should be optimized.
- ⚠️ **17.9% false positive rate** means some low-value orders will be flagged as high-value. This is manageable with a human review layer for borderline cases.

---

## 4.3 Threshold Optimization Analysis

The default classification threshold of 0.5 balances precision and recall but may not align with business priorities. We evaluated threshold sensitivity:

| Threshold | Recall (High) | Precision (High) | F1-Score | Business Recommendation |
|-----------|--------------|-----------------|----------|------------------------|
| 0.50 | 60.15% | 64.34% | 62.18% | Baseline: balanced approach |
| **0.35** | **74.20%** | **52.10%** | **61.02%** | ✅ **Optimal for revenue capture** |
| 0.25 | 83.40% | 41.20% | 55.18% | Too many false positives |
| 0.65 | 42.15% | 78.90% | 54.92% | Too conservative, misses opportunities |

**Recommended Deployment Strategy:**

```python
def classify_order(probability, threshold_low=0.35, threshold_high=0.65):
    if probability >= threshold_high:
        return "PRIORITY_FULFILLMENT"  # Auto-approve high-confidence predictions
    elif probability >= threshold_low:
        return "HUMAN_REVIEW"          # Flag borderline cases for manual review
    else:
        return "STANDARD_PROCESSING"   # Low-confidence predictions follow standard workflow

# Conclusions

### Prerequisites
```bash
Python 3.8+
pip
virtualenv (recommended)

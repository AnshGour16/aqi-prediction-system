# Detailed Model Evaluation Report

This legacy report uses a random row-wise 80/20 split. Rows from the same date can appear in both partitions, so these metrics are exploratory and are not an out-of-time evaluation or evidence of future forecasting performance. Use `python src/evaluate_temporal.py` for date-grouped holdout metrics.

## Overall Summary Table

|    | Model                            |     MAE |    RMSE |     R2 |   Cat_Accuracy |   Cat_Precision |   Cat_Recall |   Cat_F1 |
|---:|:---------------------------------|--------:|--------:|-------:|---------------:|----------------:|-------------:|---------:|
|  0 | Random Forest (Tuned)            |  0.861  |  2.9213 | 0.9996 |         0.994  |          0.994  |       0.994  |   0.994  |
|  1 | Gradient Boosting (Tuned)        |  2.2    |  3.7358 | 0.9994 |         0.929  |          0.9305 |       0.929  |   0.929  |
|  2 | Decision Tree (Tuned)            |  1.2344 |  4.2131 | 0.9992 |         0.9959 |          0.9959 |       0.9959 |   0.9959 |
|  3 | K-Nearest Neighbors (Tuned)      | 20.1977 | 27.4124 | 0.9674 |         0.8103 |          0.8431 |       0.8103 |   0.8042 |
|  4 | Support Vector Regressor (Tuned) | 12.1829 | 16.6097 | 0.988  |         0.8    |          0.8129 |       0.8    |   0.7989 |
|  5 | Ridge Regression (Tuned)         | 29.2636 | 36.7999 | 0.9412 |         0.6637 |          0.6994 |       0.6637 |   0.6583 |

## Random Forest (Tuned)

### Regression Metrics
- **MAE:** 0.8610
- **RMSE:** 2.9213
- **R² Score:** 0.9996 (99.96%)

### Derived Classification Metrics
- **Categorical Accuracy:** 99.40%
- **Weighted Precision:** 0.9940
- **Weighted Recall:** 0.9940
- **Weighted F1-Score:** 0.9940

### Confusion Matrix
|                   |   Pred Good |   Pred Satisfactory |   Pred Moderate |   Pred Poor |   Pred Very Poor |   Pred Severe |
|:------------------|------------:|--------------------:|----------------:|------------:|-----------------:|--------------:|
| True Good         |        1700 |                   7 |               0 |           0 |                0 |             0 |
| True Satisfactory |           1 |                1708 |               3 |           0 |                0 |             0 |
| True Moderate     |           0 |                   0 |            1720 |          17 |                0 |             0 |
| True Poor         |           0 |                   0 |               0 |        1662 |               15 |             0 |
| True Very Poor    |           0 |                   0 |               0 |           1 |             1633 |            16 |
| True Severe       |           0 |                   0 |               0 |           0 |                1 |          1716 |

---

## Gradient Boosting (Tuned)

### Regression Metrics
- **MAE:** 2.2000
- **RMSE:** 3.7358
- **R² Score:** 0.9994 (99.94%)

### Derived Classification Metrics
- **Categorical Accuracy:** 92.90%
- **Weighted Precision:** 0.9305
- **Weighted Recall:** 0.9290
- **Weighted F1-Score:** 0.9290

### Confusion Matrix
|                   |   Pred Good |   Pred Satisfactory |   Pred Moderate |   Pred Poor |   Pred Very Poor |   Pred Severe |
|:------------------|------------:|--------------------:|----------------:|------------:|-----------------:|--------------:|
| True Good         |        1624 |                  83 |               0 |           0 |                0 |             0 |
| True Satisfactory |           4 |                1628 |              80 |           0 |                0 |             0 |
| True Moderate     |           0 |                   0 |            1526 |         211 |                0 |             0 |
| True Poor         |           0 |                   0 |               2 |        1426 |              249 |             0 |
| True Very Poor    |           0 |                   0 |               0 |           2 |             1557 |            91 |
| True Severe       |           0 |                   0 |               0 |           0 |                2 |          1715 |

---

## Decision Tree (Tuned)

### Regression Metrics
- **MAE:** 1.2344
- **RMSE:** 4.2131
- **R² Score:** 0.9992 (99.92%)

### Derived Classification Metrics
- **Categorical Accuracy:** 99.59%
- **Weighted Precision:** 0.9959
- **Weighted Recall:** 0.9959
- **Weighted F1-Score:** 0.9959

### Confusion Matrix
|                   |   Pred Good |   Pred Satisfactory |   Pred Moderate |   Pred Poor |   Pred Very Poor |   Pred Severe |
|:------------------|------------:|--------------------:|----------------:|------------:|-----------------:|--------------:|
| True Good         |        1702 |                   5 |               0 |           0 |                0 |             0 |
| True Satisfactory |           7 |                1704 |               1 |           0 |                0 |             0 |
| True Moderate     |           0 |                   1 |            1723 |          13 |                0 |             0 |
| True Poor         |           0 |                   0 |               2 |        1673 |                2 |             0 |
| True Very Poor    |           0 |                   0 |               0 |           2 |             1641 |             7 |
| True Severe       |           0 |                   0 |               0 |           0 |                2 |          1715 |

---

## K-Nearest Neighbors (Tuned)

### Regression Metrics
- **MAE:** 20.1977
- **RMSE:** 27.4124
- **R² Score:** 0.9674 (96.74%)

### Derived Classification Metrics
- **Categorical Accuracy:** 81.03%
- **Weighted Precision:** 0.8431
- **Weighted Recall:** 0.8103
- **Weighted F1-Score:** 0.8042

### Confusion Matrix
|                   |   Pred Good |   Pred Satisfactory |   Pred Moderate |   Pred Poor |   Pred Very Poor |   Pred Severe |
|:------------------|------------:|--------------------:|----------------:|------------:|-----------------:|--------------:|
| True Good         |         759 |                 948 |               0 |           0 |                0 |             0 |
| True Satisfactory |          14 |                1548 |             150 |           0 |                0 |             0 |
| True Moderate     |           0 |                  43 |            1385 |         309 |                0 |             0 |
| True Poor         |           0 |                   0 |             125 |        1536 |               16 |             0 |
| True Very Poor    |           0 |                   0 |               0 |          69 |             1445 |           136 |
| True Severe       |           0 |                   0 |               0 |           0 |              125 |          1592 |

---

## Support Vector Regressor (Tuned)

### Regression Metrics
- **MAE:** 12.1829
- **RMSE:** 16.6097
- **R² Score:** 0.9880 (98.80%)

### Derived Classification Metrics
- **Categorical Accuracy:** 80.00%
- **Weighted Precision:** 0.8129
- **Weighted Recall:** 0.8000
- **Weighted F1-Score:** 0.7989

### Confusion Matrix
|                   |   Pred Good |   Pred Satisfactory |   Pred Moderate |   Pred Poor |   Pred Very Poor |   Pred Severe |
|:------------------|------------:|--------------------:|----------------:|------------:|-----------------:|--------------:|
| True Good         |        1166 |                 541 |               0 |           0 |                0 |             0 |
| True Satisfactory |           5 |                1390 |             317 |           0 |                0 |             0 |
| True Moderate     |           0 |                   1 |            1224 |         512 |                0 |             0 |
| True Poor         |           0 |                   0 |              33 |        1405 |              239 |             0 |
| True Very Poor    |           0 |                   0 |               0 |          22 |             1286 |           342 |
| True Severe       |           0 |                   0 |               0 |           0 |               28 |          1689 |

---

## Ridge Regression (Tuned)

### Regression Metrics
- **MAE:** 29.2636
- **RMSE:** 36.7999
- **R² Score:** 0.9412 (94.12%)

### Derived Classification Metrics
- **Categorical Accuracy:** 66.37%
- **Weighted Precision:** 0.6994
- **Weighted Recall:** 0.6637
- **Weighted F1-Score:** 0.6583

### Confusion Matrix
|                   |   Pred Good |   Pred Satisfactory |   Pred Moderate |   Pred Poor |   Pred Very Poor |   Pred Severe |
|:------------------|------------:|--------------------:|----------------:|------------:|-----------------:|--------------:|
| True Good         |         806 |                 775 |             126 |           0 |                0 |             0 |
| True Satisfactory |           0 |                 557 |            1155 |           0 |                0 |             0 |
| True Moderate     |           0 |                   0 |            1263 |         474 |                0 |             0 |
| True Poor         |           0 |                   0 |             108 |        1425 |              144 |             0 |
| True Very Poor    |           0 |                   0 |               0 |         135 |             1169 |           346 |
| True Severe       |           0 |                   0 |               0 |           6 |              161 |          1550 |

---

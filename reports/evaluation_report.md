# Detailed Model Evaluation Report

This report evaluates the tuned models on both continuous regression metrics and derived CPCB AQI classification bands on the balanced dataset.

## Overall Summary Table

|    | Model                            |     MAE |    RMSE |     R2 |   Cat_Accuracy |   Cat_Precision |   Cat_Recall |   Cat_F1 |
|---:|:---------------------------------|--------:|--------:|-------:|---------------:|----------------:|-------------:|---------:|
|  0 | Random Forest (Tuned)            |  0.7202 |  8.512  | 0.9971 |         0.9888 |          0.9889 |       0.9888 |   0.9888 |
|  1 | Gradient Boosting (Tuned)        |  2.5993 |  8.0106 | 0.9974 |         0.9631 |          0.9647 |       0.9631 |   0.9633 |
|  2 | Decision Tree (Tuned)            |  0.7674 |  9.5833 | 0.9963 |         0.9983 |          0.9983 |       0.9983 |   0.9983 |
|  3 | K-Nearest Neighbors (Tuned)      | 13.3141 | 21.0425 | 0.9823 |         0.7593 |          0.791  |       0.7593 |   0.7579 |
|  4 | Support Vector Regressor (Tuned) |  8.3841 | 15.3334 | 0.9906 |         0.8207 |          0.8296 |       0.8207 |   0.8205 |
|  5 | Ridge Regression (Tuned)         | 18.4646 | 25.7412 | 0.9735 |         0.6882 |          0.7179 |       0.6882 |   0.681  |

## Random Forest (Tuned)

### Regression Metrics
- **MAE:** 0.7202
- **RMSE:** 8.5120
- **R² Score:** 0.9971 (99.71%)

### Derived Classification Metrics
- **Categorical Accuracy:** 98.88%
- **Weighted Precision:** 0.9889
- **Weighted Recall:** 0.9888
- **Weighted F1-Score:** 0.9888

### Confusion Matrix
|                   |   Pred Good |   Pred Satisfactory |   Pred Moderate |   Pred Poor |   Pred Very Poor |   Pred Severe |
|:------------------|------------:|--------------------:|----------------:|------------:|-----------------:|--------------:|
| True Good         |        1539 |                   2 |               0 |           0 |                0 |             1 |
| True Satisfactory |           0 |                1519 |              39 |           2 |                0 |             4 |
| True Moderate     |           0 |                   0 |            1792 |          23 |                0 |             3 |
| True Poor         |           0 |                   0 |               0 |        1456 |               30 |             2 |
| True Very Poor    |           0 |                   0 |               0 |           0 |             1833 |             1 |
| True Severe       |           0 |                   1 |               1 |           1 |                4 |          1947 |

---

## Gradient Boosting (Tuned)

### Regression Metrics
- **MAE:** 2.5993
- **RMSE:** 8.0106
- **R² Score:** 0.9974 (99.74%)

### Derived Classification Metrics
- **Categorical Accuracy:** 96.31%
- **Weighted Precision:** 0.9647
- **Weighted Recall:** 0.9631
- **Weighted F1-Score:** 0.9633

### Confusion Matrix
|                   |   Pred Good |   Pred Satisfactory |   Pred Moderate |   Pred Poor |   Pred Very Poor |   Pred Severe |
|:------------------|------------:|--------------------:|----------------:|------------:|-----------------:|--------------:|
| True Good         |        1495 |                  23 |               0 |           0 |                0 |            24 |
| True Satisfactory |           0 |                1473 |              44 |           1 |                0 |            46 |
| True Moderate     |           0 |                   0 |            1752 |          40 |                0 |            26 |
| True Poor         |           0 |                   0 |               0 |        1446 |               24 |            18 |
| True Very Poor    |           0 |                   0 |               0 |          31 |             1715 |            88 |
| True Severe       |           0 |                   1 |               1 |           1 |                8 |          1943 |

---

## Decision Tree (Tuned)

### Regression Metrics
- **MAE:** 0.7674
- **RMSE:** 9.5833
- **R² Score:** 0.9963 (99.63%)

### Derived Classification Metrics
- **Categorical Accuracy:** 99.83%
- **Weighted Precision:** 0.9983
- **Weighted Recall:** 0.9983
- **Weighted F1-Score:** 0.9983

### Confusion Matrix
|                   |   Pred Good |   Pred Satisfactory |   Pred Moderate |   Pred Poor |   Pred Very Poor |   Pred Severe |
|:------------------|------------:|--------------------:|----------------:|------------:|-----------------:|--------------:|
| True Good         |        1542 |                   0 |               0 |           0 |                0 |             0 |
| True Satisfactory |           0 |                1560 |               2 |           1 |                0 |             1 |
| True Moderate     |           0 |                   0 |            1814 |           0 |                4 |             0 |
| True Poor         |           0 |                   0 |               0 |        1484 |                4 |             0 |
| True Very Poor    |           0 |                   0 |               0 |           0 |             1834 |             0 |
| True Severe       |           1 |                   1 |               0 |           3 |                0 |          1949 |

---

## K-Nearest Neighbors (Tuned)

### Regression Metrics
- **MAE:** 13.3141
- **RMSE:** 21.0425
- **R² Score:** 0.9823 (98.23%)

### Derived Classification Metrics
- **Categorical Accuracy:** 75.93%
- **Weighted Precision:** 0.7910
- **Weighted Recall:** 0.7593
- **Weighted F1-Score:** 0.7579

### Confusion Matrix
|                   |   Pred Good |   Pred Satisfactory |   Pred Moderate |   Pred Poor |   Pred Very Poor |   Pred Severe |
|:------------------|------------:|--------------------:|----------------:|------------:|-----------------:|--------------:|
| True Good         |         694 |                 804 |               9 |           0 |                0 |            35 |
| True Satisfactory |          71 |                1262 |             206 |           0 |                0 |            25 |
| True Moderate     |           0 |                 227 |            1383 |         184 |                0 |            24 |
| True Poor         |           0 |                   0 |             175 |        1256 |               37 |            20 |
| True Very Poor    |           0 |                   0 |               0 |         391 |             1381 |            62 |
| True Severe       |           1 |                   4 |               0 |           4 |              176 |          1769 |

---

## Support Vector Regressor (Tuned)

### Regression Metrics
- **MAE:** 8.3841
- **RMSE:** 15.3334
- **R² Score:** 0.9906 (99.06%)

### Derived Classification Metrics
- **Categorical Accuracy:** 82.07%
- **Weighted Precision:** 0.8296
- **Weighted Recall:** 0.8207
- **Weighted F1-Score:** 0.8205

### Confusion Matrix
|                   |   Pred Good |   Pred Satisfactory |   Pred Moderate |   Pred Poor |   Pred Very Poor |   Pred Severe |
|:------------------|------------:|--------------------:|----------------:|------------:|-----------------:|--------------:|
| True Good         |        1219 |                 263 |               0 |           0 |                0 |            60 |
| True Satisfactory |          62 |                1239 |             220 |           0 |                0 |            43 |
| True Moderate     |           0 |                  83 |            1451 |         242 |                0 |            42 |
| True Poor         |           0 |                   0 |             127 |        1218 |              113 |            30 |
| True Very Poor    |           0 |                   0 |               0 |         285 |             1334 |           215 |
| True Severe       |           2 |                   1 |               0 |           4 |               37 |          1910 |

---

## Ridge Regression (Tuned)

### Regression Metrics
- **MAE:** 18.4646
- **RMSE:** 25.7412
- **R² Score:** 0.9735 (97.35%)

### Derived Classification Metrics
- **Categorical Accuracy:** 68.82%
- **Weighted Precision:** 0.7179
- **Weighted Recall:** 0.6882
- **Weighted F1-Score:** 0.6810

### Confusion Matrix
|                   |   Pred Good |   Pred Satisfactory |   Pred Moderate |   Pred Poor |   Pred Very Poor |   Pred Severe |
|:------------------|------------:|--------------------:|----------------:|------------:|-----------------:|--------------:|
| True Good         |         592 |                 916 |               2 |           0 |                0 |            32 |
| True Satisfactory |          61 |                 667 |             796 |           0 |                0 |            40 |
| True Moderate     |           0 |                  81 |            1545 |         177 |                0 |            15 |
| True Poor         |           0 |                   0 |             189 |        1198 |               83 |            18 |
| True Very Poor    |           0 |                   0 |               0 |         348 |             1234 |           252 |
| True Severe       |           1 |                   3 |               2 |           3 |              161 |          1784 |

---

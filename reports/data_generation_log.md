# High-Quality Air Quality Dataset Generation & Audit Log

**Generated on:** 2026-11-27 11:51:50  
**Total Records:** 51000  
**Features:** 12  
**Sampling Strategy:** Stratified physical sampling across all 6 Indian CPCB AQI Categories, with one dominant pollutant per row and realistic mixed-pollutant backgrounds  

---

## 1. Class Distribution & Balance Verification

| Category | AQI Range | Sample Count | Percentage |
| :--- | :---: | :---: | :---: |
| **Good** | 0 - 50 | 8500 | 16.67% |
| **Satisfactory** | 51 - 100 | 8500 | 16.67% |
| **Moderate** | 101 - 200 | 8500 | 16.67% |
| **Poor** | 201 - 300 | 8500 | 16.67% |
| **Very Poor** | 301 - 400 | 8500 | 16.67% |
| **Severe** | 401 - 500 | 8500 | 16.67% |

---

## 2. Statistical Summary of Generated Features

|       |   pm25_median |   pm10_median |   no2_median |   so2_median |   co_median |   o3_median |   temperature_median |   humidity_median |   wind-speed_median |       AQI |
|:------|--------------:|--------------:|-------------:|-------------:|------------:|------------:|---------------------:|------------------:|--------------------:|----------:|
| count |     51000     |     51000     |    51000     |   51000      | 51000       |   51000     |          51000       |        51000      |         51000       | 51000     |
| mean  |       118.311 |       207.224 |      140.724 |     419.955  |     9.88292 |     191.052 |             22.2586  |           62.2934 |             4.02815 |   239.476 |
| std   |       115.6   |       166.271 |      134.387 |     502.678  |    11.7581  |     230.877 |              9.31483 |           16.2917 |             2.74482 |   151.659 |
| min   |         2     |         2.46  |        3     |       1      |     0.1     |       5     |             -5       |           25      |             0.2     |     9.3   |
| 25%   |        32.52  |        67.2   |       33.29  |      36.3175 |     0.94    |      37.94  |             16       |           50.5    |             1.88    |    91.8   |
| 50%   |        79.31  |       157.07  |       88.99  |     209.185  |     5.14    |      89.78  |             22.8     |           62.3    |             3.3     |   200.6   |
| 75%   |       166.175 |       326.192 |      215.25  |     647.785  |    14.6     |     214.835 |             29.2     |           74.5    |             5.56    |   384.6   |
| max   |       480     |       599.97  |      549.99  |    1999.97   |    48       |     950     |             45       |          100      |            15.16    |   500     |

---

## 3. Physical & Chemical Domain Constraints Enforced
1. **Particulate Coupling:** $PM_{10} \ge PM_{2.5}$ strictly enforced for all rows ($PM_{2.5}/PM_{10}$ ratio bounded between 0.40 and 0.85).
2. **Atmospheric Dispersion Physics:** Inversion layers and stagnant air conditions (low wind speed $< 1.5$ m/s) coupled with particulate accumulation in *Very Poor* and *Severe* categories.
3. **Photochemical Formation:** Ozone ($O_3$) levels correlated positively with ambient temperature and spring/summer seasonal context, clipped to the sampled concentration range.
4. **CPCB Breakpoint Alignment:** Contiguous linear interpolation per pollutant breakpoint curve, so no concentration can fall between bands and be mis-scored.
5. **Anchored-Pollutant Mixtures:** Each row's target category is anchored by one pollutant sampled inside the target tier, while the remaining pollutants vary from their Good-range minimum up to the same tier ceiling. This teaches the model the CPCB maximum-sub-index rule and allows realistic co-occurring pollutants.
6. **Label Consistency:** Every `Category` value is derived from its `AQI` value. Category/AQI mismatches in this dataset: **0**.

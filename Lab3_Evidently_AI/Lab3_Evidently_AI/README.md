# 📊 Evidently AI – Data Drift Analysis on Credit Dataset

This project demonstrates the use of **Evidently AI** to detect and visualize **data drift** between two subsets of the **UCI Credit dataset (OpenML ID: credit-g)**.
Data drift analysis ensures the stability of data distributions over time, which is critical for maintaining reliable and fair machine learning models in production environments.

---

## 🧠 Overview

The analysis evaluates **dataset drift** between two customer groups based on their **checking account status**:

* **Reference Dataset (`credit_ref`)** → Customers **not having low account balances** (checking status ≠ `‘<0’`, `‘0<=X<200’`)
* **Production Dataset (`credit_prod`)** → Customers **with lower account balances** (`checking_status` ∈ `['<0', '0<=X<200']`)

Evidently AI computes drift metrics for both **numerical** and **categorical** columns, identifies which features changed significantly, and visualizes their distributions to assess stability.

---

## 📦 Tools and Libraries Used

| Library                  | Purpose                           |
| ------------------------ | --------------------------------- |
| 🐍 Python3               | Core programming environment      |
| 🧮 Pandas, NumPy         | Data handling and transformation  |
| 📊 scikit-learn          | Dataset fetching (`fetch_openml`) |
| 📈 Evidently AI (v0.7.0) | Drift detection and visualization |
| 🎨 Matplotlib            | Manual feature drift plotting     |

---

## 📂 Dataset

The **German Credit (credit-g)** dataset from OpenML contains **1,000 samples** with **20+ features**, describing financial and personal attributes of loan applicants.

### Example Features

| Feature                  | Type        | Description                                      |
| ------------------------ | ----------- | ------------------------------------------------ |
| `duration`               | Numerical   | Duration of credit in months                     |
| `credit_amount`          | Numerical   | Credit amount requested                          |
| `installment_commitment` | Numerical   | Installment as a percentage of disposable income |
| `checking_status`        | Categorical | Status of existing checking account              |
| `employment`             | Categorical | Employment duration                              |
| `purpose`                | Categorical | Purpose of the loan                              |
| `age`                    | Numerical   | Applicant’s age in years                         |
| `personal_status`        | Categorical | Marital and gender status                        |
| `foreign_worker`         | Categorical | Whether the applicant is a foreign worker        |
| `class`                  | Target      | Credit risk: good/bad                            |

---

## 📊 Results and Insights

### 🧾 Dataset Drift Summary

| Metric                   | Value             |
| ------------------------ | ----------------- |
| Total Columns            | **21**            |
| Drifted Columns          | **15**            |
| Share of Drifted Columns | **0.714 (71.4%)** |
| Drift Threshold          | **0.5**           |
| Dataset Drift            | ✅ **Detected**    |

> Drift is detected for **71.43%** of columns, indicating substantial distributional differences between the reference and production datasets.

---

### ⚖️ Top Drifted Features

| Feature               | Type        | Stat Test  | p-value | Drift      |
| --------------------- | ----------- | ---------- | ------- | ---------- |
| `job`                 | Categorical | Chi-square | 0.006   | ✅ Detected |
| `personal_status`     | Categorical | Chi-square | 0.017   | ✅ Detected |
| `other_payment_plans` | Categorical | Chi-square | 0.021   | ✅ Detected |
| `age`                 | Numerical   | KS Test    | 0.0008  | ✅ Detected |
| `purpose`             | Categorical | Chi-square | 0.032   | ✅ Detected |

### 🧮 Non-Drifted Features (Stable)

| Feature                  | Type        | Drift          |
| ------------------------ | ----------- | -------------- |
| `foreign_worker`         | Categorical | ❌ Not Detected |
| `num_dependents`         | Numerical   | ❌ Not Detected |
| `installment_commitment` | Numerical   | ❌ Not Detected |

---

## 💡 Key Takeaways

1. **Substantial drift detected** in 71% of features — notably demographic (`age`, `personal_status`) and financial (`job`, `purpose`) attributes.
2. **Behavioral & financial changes** in the population might be affecting model input distribution.
3. **Stable attributes** such as `foreign_worker` and `installment_commitment` indicate certain consistent demographic and financial behavior patterns.
4. **Chi-square and KS tests** were applied automatically by Evidently to quantify the distributional shift.

---

## 🚀 Future Improvements

* Integrate drift monitoring into an **MLOps pipeline** for automated checks.
* Add **target drift** and **model performance monitoring**.
* Use **Evidently Cloud or Airflow tasks** to schedule periodic drift evaluations.
* Combine with **alert systems** (e.g., Slack, email) for early detection of significant shifts.

---

## 🧾 Conclusion

This Evidently AI analysis successfully identified **significant data drift** across a majority of features in the Credit dataset.
The results emphasize the importance of **continuous dataset monitoring** to ensure that models remain fair, accurate, and generalizable in changing environments.

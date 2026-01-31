# Student-Dropout-Early-WarningSystem
# Student Dropout Early Warning System – Short Report

## 1. Data Cleaning

- **Removed extra spaces** in column names using `df.columns.str.strip()`  
- **Checked for missing values**; dataset was complete, so no imputation needed  
- **Converted target variable** `Class` into binary `dropout`:
  - `L` → 1 (Dropout)  
  - `H` → 0 (Continue)  
- Ensured **categorical columns** are consistent with one-hot encoding during model training  

---

## 2. Features Used

| Type | Features |
|------|----------|
| Categorical | gender, StageID, GradeID, Topic, Semester, StudentAbsenceDays, Relation |
| Numerical   | raisedhands, VisITedResources, AnnouncementsView, Discussion |

- Total 11 features used for prediction  
- Selected based on **student engagement and academic activity**, focusing on early-semester indicators  

---

## 3. Model Choice & Metrics

- **Model:** Logistic Regression (balanced class weights, max_iter=1000)  
- **Pipeline:** Preprocessing (StandardScaler for numeric + OneHotEncoder for categorical) + Logistic Regression classifier  
- **Evaluation Metrics:**
  - Classification Report (precision, recall, f1-score)
  - ROC-AUC Score (to measure model's ability to distinguish dropout vs continue)  

- **Reason for choice:** Logistic Regression is **simple, interpretable, and easy for advisors to understand**  

---

## 4. Risk Thresholds

- Risk Score ranges from 0 to 1 (probability of dropout)  
- **High Risk:** risk_score ≥ 0.7  
- **Medium Risk:** 0.4 ≤ risk_score < 0.7  
- **Low Risk:** risk_score < 0.4  
- Predicted dropout = risk_score ≥ 0.5  

> Thresholds set to **catch most dropout students early**, while **keeping false alarms reasonable**  

---

## 5. Key Reasons Behind Dropout Predictions

- **Low Class Participation:** Few raised hands in class activities  
- **Low Engagement with Learning Resources:** Minimal access to course materials (`VisITedResources`)  
- **High Absences:** Student absent for multiple days early in the semester  
- **Limited Discussions:** Low participation in discussion forums / announcements  
- **Other factors:** Poor engagement combined with low grades or academic stage  

> These indicators allow advisors to **proactively reach out to at-risk students** before it is too late  

---

## 6. Conclusion

This system provides **early warning for student dropout** using easily interpretable metrics. Advisors can use:

- Risk scores and labels (Low / Medium / High)  
- Top 20 high-risk student lists  
- Individual student reports  

This allows for **timely intervention and targeted support** to reduce student dropout rates.

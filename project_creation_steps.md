# COM763 Task 1 - Diabetes Risk Screening

This guide gives you a clear step-by-step path to complete Task 1 with your chosen topic: **Diabetes Risk Screening**.

You asked for:
- 2000 words report
- Same report structure
- Easy and practical workflow

This file is made to match that exactly.

---

## 1) Project Goal (keep this fixed)

**Problem statement:**
Build a machine learning system that predicts whether a person is at risk of diabetes using health indicators.

**Type of ML task:**
Binary classification (At Risk / Not At Risk).

**Real-world use:**
A screening support tool for early health awareness.

**Important note for report and app:**
This system is for educational screening support only, not a medical diagnosis.

---

## 2) Create Your Project Files (local + GitHub)

Inside this folder, keep this structure:

```
diabetes_risk_project/
  Student_Project.ipynb
  app.py
  model.pkl
  dataset.csv
  requirements.txt
  report_assets/
    figures/
    screenshots/
```

### Why this matters
- This matches the lecturer workflow guide.
- It helps you avoid common mistakes (missing ipynb, missing model.pkl, missing requirements).

---

## 3) Dataset Selection and Save

Use the Pima Indians Diabetes dataset (CSV format).

Actions:
1. Download the dataset CSV.
2. Rename it to `dataset.csv`.
3. Save it in this folder.
4. Keep a short note of dataset source for references section.

---

## 4) Build Notebook in 6 Practical Sections

Create `Student_Project.ipynb` and build in this order:

1. Load dataset
2. Clean data
3. EDA and visuals
4. Train models
5. Evaluate and compare
6. Save final model as `model.pkl`

Do not skip section 6.

### Section details

### 4.1 Load dataset
- Import pandas, numpy, matplotlib, seaborn, sklearn.
- Read `dataset.csv`.
- Show shape, column names, and first few rows.

### 4.2 Clean data
- Check missing values.
- In this dataset, some zero values are invalid for medical features (for example glucose, BMI).
- Replace invalid zeros with NaN where needed.
- Impute values (median is a safe simple choice).
- Remove duplicates if found.

### 4.3 EDA and visuals
Create clear visuals for report:
- Class distribution bar chart
- Correlation heatmap
- Histograms for key features
- Boxplots for outlier understanding

Save figures into `report_assets/figures/`.

### 4.4 Train models
Use train/test split and a pipeline.
Try at least 3 models:
1. Logistic Regression (baseline)
2. Random Forest
3. Gradient Boosting (or XGBoost if available)

If needed, scale numeric features for Logistic Regression.

### 4.5 Evaluate and compare
For each model, calculate:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Make one comparison table and one chart for easy reading.

### 4.6 Save final model
- Choose best model based on evidence.
- Save it as `model.pkl` using pickle.

---

## 5) Debugging Evidence (important for marks)

Your report needs proof of iterative improvement.

Keep short notes while coding:
- What issue happened
- Why it happened
- How you fixed it
- What changed in results

Example debugging notes:
1. Class imbalance reduced recall -> used class_weight or threshold tuning.
2. Overfitting in Random Forest -> tuned max_depth and min_samples_leaf.
3. Poor baseline performance -> improved preprocessing and feature handling.

Take screenshots of key notebook cells and outputs.
Save to `report_assets/screenshots/`.

---

## 6) Create Streamlit App

Create `app.py` with:
1. Load `model.pkl`
2. Input fields for health values
3. Predict button
4. Output risk result
5. Short disclaimer text

### Recommended input fields
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

### Output style
- Predicted class: At Risk / Not At Risk
- Optional probability score (for transparency)

---

## 7) requirements.txt

Create this file and include libraries you used. Example:

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
```

If you use xgboost, add:

```
xgboost
```

---

## 8) Push to GitHub (Public)

Actions:
1. Create a new repository (public).
2. Upload all required files.
3. Confirm these exist in repo root:
   - Student_Project.ipynb
   - app.py
   - model.pkl
   - dataset.csv
   - requirements.txt

Keep repo public so lecturer can access it.

---

## 9) Deploy on Streamlit Cloud

Actions:
1. Go to Streamlit Community Cloud.
2. Connect GitHub account.
3. Select repository.
4. Select `app.py` as entry file.
5. Deploy.
6. Copy public Streamlit URL.

Test with 3 to 5 different input combinations and capture screenshots.

---

## 10) Report Structure with 2000-Word Plan

You want to follow your given structure and still reach 2000 words. Use this adjusted target while keeping the same section headings.

### Final section plan
1. Introduction & problem definition - **250 words**
2. Data description & EDA - **500 words**
3. Model implementation & debugging - **550 words**
4. Evaluation & model comparison - **500 words**
5. Conclusion & reflection - **200 words**
6. Streamlit URL - not counted
7. References - usually not counted

Main body total = **2000 words**

---

## 11) What to Write in Each Report Section

## 11.1 Introduction & problem definition (250)
Write:
- Real-world diabetes burden in simple terms
- Why ML helps screening
- Your objective and scope
- Clear output definition (At Risk / Not At Risk)
- One line disclaimer (not diagnosis)

## 11.2 Data description & EDA (500 + visuals)
Write:
- Dataset source and size
- Feature descriptions
- Cleaning steps and justification
- Key EDA findings with visuals
- Brief comments on patterns (for example higher glucose linked with risk)

## 11.3 Model implementation & debugging (550 + screenshots)
Write:
- Train/test split setup
- Pipeline design
- Why each model was selected
- Hyperparameter tuning steps
- 2 to 3 debugging examples with evidence
- Screenshots of important code blocks only (not full notebook dump)

## 11.4 Evaluation & model comparison (500 + tables/charts)
Write:
- Metrics and why they matter
- Comparison table for all models
- Confusion matrix and ROC-AUC discussion
- Why final model was selected
- Trade-off discussion (for example recall vs precision)

## 11.5 Conclusion & reflection (200)
Write:
- What worked well
- What was challenging
- Practical limitations
- Future improvements (for example more data, threshold tuning)

## 11.6 Streamlit URL
Add the deployed app link clearly.

## 11.7 References
Include dataset source, library docs, and any paper/book sources used.

---

## 12) Evidence Checklist Before Submission

You should have all of these ready:

1. `Student_Project.ipynb` complete and clean
2. `model.pkl` saved and tested
3. `app.py` working locally and on Streamlit
4. `requirements.txt` complete
5. Public GitHub repo URL
6. Public Streamlit URL
7. Report with required visuals and screenshots
8. Word count near 2000 (within allowed range)
9. References added
10. Final proofreading for human and clear tone

---

## 13) 3-Day Fast Completion Plan

### Day 1
- Finalize notebook sections 1 to 4
- Run baseline and 2 more models
- Save figures and screenshots

### Day 2
- Finish evaluation and model selection
- Save `model.pkl`
- Build and test `app.py`
- Push everything to GitHub

### Day 3
- Deploy on Streamlit
- Write report using section word targets
- Add links and final checks
- Submit before deadline

---

## 14) Common Mistakes to Avoid

1. Forgetting to save `model.pkl`
2. Missing `requirements.txt`
3. Private GitHub repository
4. Missing Streamlit URL in report
5. Too many code dumps instead of key screenshots
6. Weak explanation of why final model was selected

---

This guide is your execution plan. Follow it step by step and you can complete Task 1 in a clean, mark-friendly way.

# 🎓 Student Performance Predictor

This is a machine learning web app built with **Streamlit** that predicts a student's academic performance category — **Low**, **Medium**, or **High** — based on their demographic and educational background.

---

## 📊 Dataset

The dataset is sourced from Kaggle:  
**[Student Performance in Mathematics Dataset](https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics)**

Each record includes features such as:
- Parental level of education
- Lunch type
- Test preparation course
- Gender
- Race/Ethnicity
- Scores in Math, Reading, and Writing

---

## 🧠 Model Details

- **Model:** `RandomForestClassifier`
- **Preprocessing:**
  - **Ordinal Encoding** for ordered categorical features
  - **One-Hot Encoding** for nominal features
  - **Standard Scaling** on average score
- **Target:** `performance_level`, classified into:
  - `Low`: average score < 70
  - `Medium`: 70 ≤ average score < 85
  - `High`: average score ≥ 85

---

## 🚀 How to Run Locally

### 🔧 Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

# Student Performance Predictor

A Streamlit app to predict student performance.

🌐 **Deployed app:** [https://student-performance-predictor-2059.streamlit.app/](https://student-performance-predictor-2059.streamlit.app/)


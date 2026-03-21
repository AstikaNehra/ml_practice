# 🤖 ML Interview Prep

A structured collection of Machine Learning concepts, notes, and interview questions — built while actively studying and practicing.

---

## 📚 Topics Covered

| Topic | Status |
|-------|--------|
| Linear Regression | ✅ In Progress |
| Logistic Regression | 🔜 Coming Soon |
| Decision Trees & Random Forest | 🔜 Coming Soon |
| SVM | 🔜 Coming Soon |
| Unsupervised Learning | 🔜 Coming Soon |
| Neural Networks | 🔜 Coming Soon |
| Statistics & Probability | 🔜 Coming Soon |

---

## 📁 Folder Structure

```
ml-interview-prep/
│
├── README.md
│
├── linear_regression/
│   ├── notes.md                  # Core concepts & theory
│   ├── gradient_descent.md       # GD, learning rate, derivatives
│   ├── regularization.md         # Ridge, Lasso, ElasticNet
│   └── scenario_questions.md     # Real-world scenario Q&A
│
├── logistic_regression/
│   └── notes.md
│
├── trees/
│   └── notes.md
│
├── unsupervised/
│   └── notes.md
│
└── stats_concepts/
    └── notes.md
```

---

## 🧠 Linear Regression — Quick Reference

### Key Equations

| Concept | Formula |
|---------|---------|
| Prediction | `ŷ = θ₀ + θ₁x₁ + ... + θₙxₙ` |
| MSE Loss | `J(θ) = (1/2m) Σ(ŷ - y)²` |
| Gradient Descent | `θⱼ := θⱼ - α · ∂J/∂θⱼ` |
| R² | `1 - SS_res/SS_tot` |
| OLS Solution | `θ = (XᵀX)⁻¹Xᵀy` |
| VIF | `1 / (1 - R²ⱼ)` |

### Assumptions (LINE)
- **L**inearity
- **I**ndependence of residuals
- **N**ormality of residuals
- **E**qual variance (Homoscedasticity)

### Regularization
| Method | Penalty | Feature Selection? |
|--------|---------|-------------------|
| Ridge (L2) | `λΣθ²` | ❌ No |
| Lasso (L1) | `λΣ\|θ\|` | ✅ Yes |
| ElasticNet | L1 + L2 | ✅ Partial |

---

## 💡 Interview Question Categories

### Tier 1 — Conceptual
- What is Linear Regression?
- What is R² vs Adjusted R²?
- What are the 4 assumptions?
- What is Multicollinearity?
- What is Homoscedasticity?
- What are residuals?

### Tier 2 — Advanced
- OLS closed form vs Gradient Descent
- Why Lasso does feature selection but Ridge doesn't
- Geometric intuition of L1 vs L2
- What is ElasticNet and when to use it
- Effect of lambda in regularization

### Tier 3 — Scenario Based
- Model works in training but fails in production
- High R² but business says model is useless
- Two features with 0.99 correlation
- Coefficients flipped sign after adding new feature
- 500 features, 200 rows — what do you do?
- When to prefer LR over XGBoost

---

## 🔑 Key Concepts to Remember

> **Residual** = y - ŷ (individual error per point)

> **Bias** = mean(y - ŷ) (systematic error across all points)

> **R²** never decreases when adding features — **Adjusted R²** penalizes useless ones

> **VIF > 10** → severe multicollinearity → drop or regularize

> **Homoscedasticity** = same error spread across all X values (good ✅)

> **Lasso** can make coefficients exactly 0 → feature selection (diamond geometry)

> **Ridge** only shrinks coefficients, never to exactly 0 (circle geometry)

---

## 🛠️ Setup

```bash
git clone https://github.com/YOUR_USERNAME/ml-interview-prep.git
cd ml-interview-prep
```

---

## 📝 Notes

- Notes are written while actively learning — may be updated frequently
- Focuses on **intuition first**, math second
- Interview questions sourced from real ML interview experiences

---

*Built while preparing for ML interviews 🚀*

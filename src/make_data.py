import pandas as pd
import numpy as np

# set random seed for reproducibility
rng = np.random.default_rng(42)

N = 500  # number of fake patients

ages = rng.integers(18, 90, N)
sex = rng.choice(["M", "F"], N)
has_diabetes = rng.integers(0, 2, N)
has_htn = rng.integers(0, 2, N)
prior_adm = rng.poisson(0.8, N)
avg_glucose = np.round(rng.normal(120 + 40 * has_diabetes, 20), 1)
meds_count = np.clip(np.round(rng.normal(3 + has_diabetes + has_htn, 1.2)), 0, 12).astype(int)
los = np.clip(np.round(np.random.normal(3 + 1.5 * has_htn + 0.5 * prior_adm, 1.5)), 1, 21).astype(int)

# label: readmission within 30 days
z = (
    -3.0
    + 0.02 * ages
    + 0.9 * has_diabetes
    + 0.8 * has_htn
    + 0.35 * prior_adm
    + 0.01 * (avg_glucose - 120)
    + 0.08 * meds_count
    + 0.07 * los
)
prob = 1 / (1 + np.exp(-z))
y = rng.binomial(1, prob)

df = pd.DataFrame({
    "age": ages,
    "sex": sex,
    "has_diabetes": has_diabetes,
    "has_hypertension": has_htn,
    "prior_admissions": prior_adm,
    "avg_glucose": avg_glucose,
    "meds_count": meds_count,
    "length_of_stay": los,
    "readmitted_30d": y,
})

df.to_csv("data/synthetic_ehr.csv", index=False)
print("✅ synthetic_ehr.csv created in data/")

# Piezo.AI: Accelerating Lead-Free Piezoelectric Material Discovery via Machine Learning

**Author:** Aryan  
**Affiliation:** Piezoelectrics AI Discovery Lab

---

## 1. Abstract
The transition to lead-free electronics is hindered by the slow pace of material discovery. We present **Piezo.AI**, a machine learning platform that predicts the Piezoelectric Coefficient ($d_{33}$) and Curie Temperature ($T_c$) of complex ceramics with high accuracy ($R^2 > 0.85$). By leveraging physics-based descriptors and ensemble learning, we accelerate the screening of Potassium-Sodium Niobate ($KNN$)-based solid solutions.

---

## 2. Problem Statement: The "Lead Dilemma"
-   **Context**: Lead Zirconate Titanate ($PZT$) dominates the market but contains toxic lead ($>60\%$), facing bans under **RoHS** regulations.
-   **Challenge**: Finding a lead-free replacement involves navigating an infinite compositional space (e.g., doping $KNN$ with $Li, Ta, Sb, Zr, Hf$).
-   **Bottleneck**: Traditional synthesis and characterization take weeks per sample.

**Goal:** Create a "Virtual Laboratory" to screen thousands of compositions in seconds.

---

## 3. Methodology & Approach

### A. Dataset Curation
-   **Source**: Curated dataset of **~250** experimental ferroelectric ceramics.
-   **Structure**: Perovskite solid solutions (e.g., $(K_{0.5}Na_{0.5})NbO_3$-based).
-   **Targets**:
    1.  **$d_{33}$ (pC/N)**: Large Signal Piezoelectric Coefficient.
    2.  **$T_c$ (°C)**: Curie Temperature (Phase Transition).

### B. Feature Engineering
We map chemical formulas to machine-readable vectors using **28 distinct features**:
1.  **Elemental Fractions (24 Features)**:
    Quantifies the stoichiometric ratio of elements present in the compound:
    *   *Ag, Al, B, Ba, Bi, C, Ca, Fe, Hf, Ho, K, Li, Mn, Na, Nb, O, Pr, Sb, Sc, Sr, Ta, Ti, Zn, Zr*
2.  **Physics-Based Descriptors (4 Features)**:
    We compute the stoichiometry-weighted average of fundamental atomic properties to capture chemical intuition:
    *   **Atomic Mass**: Influence on lattice vibration modes.
    *   **Atomic Radius**: Critical for Goldschmidt tolerance factor and lattice distortion.
    *   **Electronegativity (Pauling)**: Determines bond character (ionic vs. covalent).
    *   **Valence Electrons**: Dictates charge carrier concentration and bonding.

### C. Advanced Machine Learning Pipeline
Our engine ([ml_engine.py](file:///Users/lakhanprasadsahu/Documents/Projects/Piezoelectrics-AI-Discovery-Lab/backend/ml_engine.py)) employs a sophisticated multi-stage architecture:
1.  **Data Cleaning**: Recursive Regex parsing for complex nested formulas (e.g., `0.96(K0.5Na0.5)NbO3-0.04...`).
2.  **Smart Imputation**: **KNN-based Imputation** uses chemical similarity (feature vector distance) to estimate missing experimental target values in the training set.
3.  **Model Zoo**: We train and optimize 7 robust algorithms:
    -   *Tree Ensembles*: Random Forest, XGBoost, LightGBM, Gradient Boosting.
    -   *Kernel Methods*: Support Vector Regression (SVR), Kernel Ridge (KRR) with **Target Scaling**.
    -   *Probabilistic*: Gaussian Process Regression (GPR) with Matern Kernels.
4.  **Stacked Generalization**: A **Gradient Boosting Meta-Learner** aggregates predictions from all base models to minimize bias and variance.

---

## 4. Results & Performance

*(Suggested Layout: Place the table on the left and R2 Bar Charts on the right)*

### Model Evaluation Metrics
| Target Property | Best Model | $R^2$ Score | RMSE |
| :--- | :--- | :--- | :--- |
| **Piezoelectric Coeff. ($d_{33}$)** | **Stacked Ensemble** | **0.85** | **~25 pC/N** |
| **Curie Temperature ($T_c$)** | **Random Forest** | **0.90** | **~15 °C** |

### Key Insights
-   **Physics Matters**: Including electronegativity and atomic radius significantly boosted $T_c$ prediction accuracy, highlighting the importance of structural distortion in phase transitions.
-   **Ensemble Power**: The Stacking Regressor consistently outperformed single best models by 3-5%, effectively handling the non-linearities of doping effects.
-   **Compositional Drivers**: Feature importance analysis reveals that **Antimony ($Sb$)** and **Tantalum ($Ta$)** concentrations are the top predictors for tuning $T_c$.

---

## 5. Deployment: The Piezo.AI Platform
We deployed the model as a responsive web application to democratize access:
-   **Backend**: **FastAPI** (Python) for sub-millisecond inference and asynchronous tasks.
-   **Frontend**: **React.js** + **Material UI** for an intuitive, scientific dashboard.
-   **Key Features**:
    -   **Formula Builder**: Validates charge neutrality in real-time.
    -   **Active Learning**: Users can upload new data to retrain and improve the model automatically.
    -   **PDF Reporting**: One-click generation of experimental reports.

---

## 6. Conclusion & Future Work
**Piezo.AI** demonstrates that Materials Informatics can effectively bypass the trial-and-error bottleneck. By accurately predicting functional properties from simple chemical text, we empower researchers to focus experimental efforts only on the most promising lead-free candidates.

**Future Roadmap:**
1.  **Generative Design**: Using VAEs/GANs to *generate* new formulas rather than just screening them.
2.  **Structure-Property**: Integrating Crystal Graph Convolutional Networks (CGCNN) for lattice-aware predictions.

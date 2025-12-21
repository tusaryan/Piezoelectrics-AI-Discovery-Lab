# Project Report: AI-Assisted Discovery of New Lead-Free Piezoelectrics

**Project Name:** Piezo.AI Discovery Lab  
**Date:** December 16, 2025

---

## DECLARATION
I hereby declare that the work presented in this report entitled **"Piezo.AI Discovery Lab: AI-Assisted Discovery of New Lead-Free Piezoelectrics"** is the outcome of my own research and development work. This project was undertaken to democratize the discovery of sustainable materials using advanced Materials Informatics.

## ABSTRACT
The **Piezo.AI Discovery Lab** project addresses the critical environmental challenge of replacing toxic lead-based piezoelectric materials (like PZT) with eco-friendly alternatives. By developing a comprehensive **Materials Informatics** workflow, this project accelerates the discovery of high-performance lead-free ceramics, specifically focusing on Potassium-Sodium Niobate (KNN)-based solid solutions. The system leverages a curated dataset of ferroelectric ceramics to train advanced machine learning models (LightGBM, XGBoost, Random Forest, SVM, and Ensembles). These models predict two key functional properties—**Piezoelectric Coefficient ($d_{33}$)** and **Curie Temperature ($T_c$)**—directly from chemical composition, reducing screening time from weeks to milliseconds. The platform is deployed as a user-friendly web application with a dual-mode training interface (Manual vs. Auto-Intelligent), enabling researchers to perform instant predictions, comparisons, and automated report generation.

## ACKNOWLEDGEMENT
I would like to express my gratitude to the open-source community for providing the robust libraries (Scikit-learn, XGBoost, React, FastAPI) that made this research possible. Special thanks to the materials science researchers whose experimental data formed the backbone of this study.

## LIST OF FIGURES & TABLES
- **Figure 1:** Model Architecture Diagram (Dual-Model Strategy)
- **Figure 2:** Accuracy Comparison Bar Charts ($d_{33}$ & $T_c$)
- **Figure 3:** Parity Plots (Predicted vs. Actual Values)
- **Table 1:** Performance Metrics ($R^2$, RMSE) for all Algorithms

---

## 1. Introduction

### 1.1 Background: Piezoelectricity and PZT
Piezoelectric materials are the unseen workhorses of modern technology, converting mechanical stress into electrical energy and vice-versa. They are ubiquitous in medical ultrasound imaging, precision actuators, sonar, and energy harvesting devices.

### 1.2 The "Lead Dilemma" and Environmental Regulations
The industry standard, **Lead Zirconate Titanate (PZT)**, possesses excellent electromechanical properties but contains over **60% lead (Pb)** by weight. Lead is a potent neurotoxin, posing severe risks during processing and disposal. Global regulations, such as the EU's **RoHS** (Restriction of Hazardous Substances) and **REACH**, are driving an urgent deadline to phase out lead-based electronics, creating a massive demand for high-performance lead-free alternatives.

### 1.3 Materials Informatics: A New Paradigm
Traditional trial-and-error discovery is too slow to meet this demand. **Materials Informatics**—the application of data science to materials problems—offers a solution. by learning the complex, non-linear relationships between chemical composition and physical properties, AI models can virtually screen thousands of candidates in seconds, guiding experimentalists toward the most promising compositions.

## 2. LITERATURE REVIEW

### 2.1 Lead-Free Alternatives: The KNN System
Among lead-free candidates, **Potassium-Sodium Niobate ($(K,Na)NbO_3$ or KNN)** involves complex doping strategies with elements like Li, Ta, Sb, and Zr to enhance its piezoelectric response. However, the compositional space is vast and highly non-linear, making it difficult to optimize using traditional phase diagram explorations.

### 2.2 Machine Learning in Materials Science
Recent studies have demonstrated the efficacy of Support Vector Machines (SVM) and Random Forests in predicting material properties. However, most existing tools are "black boxes" accessible only to coding experts, lacking user-friendly interfaces or the ability to dynamically retrain models/datasets.

### 2.3 Existing Gaps and Limitations
1.  **Accessibility**: Lack of tools for non-coding experimentalists.
2.  **Rigidity**: Most models are static and cannot be updated with new lab data.
3.  **Transparency**: Limited explainability regarding model choices and hyperparameter impact.

## 3. OBJECTIVE
The primary objective of the **Piezo.AI Discovery Lab** is to build a full-stack, AI-driven platform that:
1.  **Accelerates Discovery**: Predicts $d_{33}$ and $T_c$ instantly for complex solid solutions.
2.  **Democratizes AI**: Provides a clean, modern UI for non-tech users with "i-button" guides for complex ML parameters.
3.  **Empowers Users**: Offers complete control via a **Dual-Model Strategy** (Manual Custom Tuning vs. Automatic Intelligent Tuning).
4.  **Ensures Robustness**: Automatically selects the best-performing algorithms based on rigorous statistical validation ($R^2$ score).

## 4. EXPERIMENTAL PROCEDURE

### 4.1 Data Collection and Curation
A dataset of **256 distinct ferroelectric ceramic compositions** was curated from experimental literature. The data includes the chemical formula string and the target functional properties ($d_{33}$ and $T_c$).
- **Imputation**: An **Advanced KNN Imputer** fills missing target values based on chemical similarity (nearest neighbors in compositional space) to maximize data utility.

### 4.2 Feature Engineering (Chemical Parsing)
A custom **Hybrid Parsing Engine** was developed to transform raw chemical text into machine-learnable vectors. This system integrates the **`chemparse` library** with a specialized **Recursive Regex layer**:
- **Custom Regex Logic**: Deconstructs complex solid solution notations (e.g., `0.96(K0.5Na0.5)NbO3-0.04...`) which standard parsers cannot handle, grouping nested brackets and coefficients.
- **Chempase Integration**: The **`chemparse`** library is the utilized to accurately parse the stoichiometry of the resolved constituent parts.
- **Elemental Descriptors (24 Features)**: Fractional composition of key elements (Ag, Al, Ba, Bi, Li, Ta, etc.).
- **Physics-Based Descriptors**: Weighted averages of fundamental atomic properties:
    - **Atomic Mass**
    - **Atomic Radius** (critical for tolerance factors)
    - **Electronegativity** (bond character)
    - **Valence Electron Count**

### 4.3 Machine Learning Architecture (Dual-Model Strategy)
The core engine provides users with two distinct pathways to train production models:

#### A. Automatic Intelligent Model Tuning ("Auto-Mode")
- **Logic**: The system runs a comprehensive benchmark across multiple algorithms to find the global optimum.
- **Model Zoo**: Random Forest, XGBoost, LightGBM, Gradient Boosting, SVM (SVR), Kernel Ridge, and Gaussian Process.
- **Selection Criteria**: The system automatically compares the **$R^2$ score** (Coefficient of Determination) of all models. It logically creates a **Stacked Ensemble** or selects the single best performing model for each property ($d_{33}$ and $T_c$) independently.

#### B. Manual Custom Tuning
- **Logic**: Gives experts granular control to test specific hypotheses.
- **Granular Control**: Users can choose specific models (e.g., "Use XGBoost for $d_{33}$ but SVR for $T_c$").
- **Parameter Guides**: The UI includes **"i-button" tooltips** explaining complex hyperparameters (e.g., *Learning Rate*, *Tree Depth*, *Epsilon*) in plain English, helping non-tech users adjust for best accuracy.

### 4.4 Software Implementation Stack
- **Backend**: Python (FastAPI) for high-performance async processing, Scikit-learn/XGBoost/LightGBM for ML.
- **Frontend**: React.js with Material UI for a polished, responsive experience.
- **Visualization**: Recharts for interactive plotting and Matplotlib/ReportLab for PDF report generation.
- **Containerization**: Docker for replicable deployment.

## 5. RESULTS AND DISCUSSION

### 5.1 Model Evaluation Metrics
Models were evaluated using an 80/20 train-test split. The **Stacked Ensemble** and **Gradient Boosting** variants consistently emerged as top performers for the complex multi-element datasets.

### 5.2 Parity Plots and Accuracy Analysis
The automated report generates **Parity Plots** (Predicted vs. Actual) which show a tight clustering of data points along the diagonal ($y=x$) line, validating the model's ability to generalize to unseen test data. The **R² values** (typically >0.85 for $d_{33}$) indicate a strong correlation, while **RMSE** provides a measure of the average error magnitude.

### 5.3 Metallurgical Insights (Feature Importance)
The backend engine calculates feature importance, identifying that **Elemental Fractions** (specifically specific dopants like Ta and Sb) and **Atomic Radius** play the most significant roles in determining $T_c$, aligning with established physical theories about lattice distortion and phase transition temperatures.

### 5.4 Web Application Deployment
The final deployed application successfully lowers the barrier to entry for AI adoption:
- **Clean & Modern UI**: A minimalist dashboard focuses on the science, hiding code complexity.
- **Interactive Dataset Viewer**: Users can view the complete dataset, upload new CSVs, and trigger retraining.
- **Prediction Section**: Allows instant "What-if" analysis for new formulas.
- **Report Generation**: A one-click export feature produces comprehensive PDF reports covering all visuals, metrics, and comparisons ("show all visuals with relevant example"), facilitating easy documentation of research progress.

## 6. CONCLUSION
The Piezo.AI Discovery Lab successfully demonstrates that data-driven approaches can significantly accelerate materials discovery. By providing a flexible **Dual-Model** architecture, the platform caters to both novice users (via Auto-Mode) and domain experts (via Manual Mode). The integration of rigorous physics-based feature engineering with state-of-the-art algorithms like LightGBM and Stacking ensures high predictive accuracy, paving the way for the rapid identification of sustainable lead-free piezoelectrics.

## 7. FUTURE SCOPE
- **Inverse Design**: Implementing Generative AI (VAEs/GANs) to generate formulas from desired properties.
- **Crystal Structure Integration**: Incorporating XRD patterns or CIF files for structurally-aware predictions.
- **Active Learning Loops**: Suggesting specific new experiments to efficiently reduce model uncertainty.

## 8. REFERENCE
1.  *Lookman, T., et al.* "Information science for materials discovery and design." *Nature Reviews Materials* (2019).
2.  *Rajan, K.* "Materials Informatics: The Materials Gene and Big Data." *Annual Review of Materials Research* (2015).
3.  *Pedregosa, F., et al.* "Scikit-learn: Machine Learning in Python." *JMLR* (2011).

---

## 9. UPDATES LOG

### **Version 1.0.0 - Initial Release (December 2025)**
- **Core Engine**: Implemented `ml_engine.py` with support for Random Forest, XGBoost, LightGBM, Gradient Boosting, SVR, KRR, and Gaussian Process.
- **Dual-Model Strategy**:
    - Added **Auto-Intelligent Tuning** (Standard & Accuracy modes).
    - Added **Manual Configuration** with independent $d_{33}$/$T_c$ model selection.
- **UI/UX**:
    - Launched React-based SPA with Material UI.
    - Implemented **"i-button" Tooltips** for hyperparameter education (e.g., n_estimators, learning_rate).
    - Added real-time training progress bars and console logs.
- **Report Generation**:
    - Automated PDF generation with `report_generator.py`.
    - Includes dynamic Bar Charts (R²/RMSE) and Scatter Plots (Predicted vs Actual).
- **Data Pipeline**:
    - Integrated `chemparse` with custom regex for complex nested formula parsing.
    - Implemented KNN-based imputation for missing target values.

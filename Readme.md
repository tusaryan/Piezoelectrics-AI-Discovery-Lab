# 🧪 AI-Assisted Discovery of New Lead-Free Piezoelectrics

![Project Status](https://img.shields.io/badge/Status-Active_Development-green)
![Tech Stack](https://img.shields.io/badge/Stack-FastAPI_React_Docker-blue)
![Domain](https://img.shields.io/badge/Domain-Materials_Informatics-purple)

## 📖 Executive Summary

This project captures a complete **Materials Informatics Workflow**, designed to solve one of the most pressing challenges in materials science: finding eco-friendly alternatives to toxic lead-based electronics.

By engineering a full-stack **"Virtual Laboratory"**, this application accelerates the discovery of lead-free piezoelectric ceramics. It replaces months of costly, trial-and-error synthesis with **instant, machine-learning-driven predictions**, enabling researchers to screen thousands of complex chemical compositions in seconds.

**Key Impact:**
*   **Acceleration:** Reduces material screening time from **weeks to milliseconds**.
*   **Precision:** Achieves high predictive accuracy ($R^2 \approx 0.85$) for critical properties like Piezoelectric Coefficient ($d_{33}$) and Curie Temperature ($T_c$).
*   **Usability:** Democritizes advanced ML models for non-computational experimentalists via an intuitive GUI.

---

## 🌍 The Problem: The "Lead Dilemma"

**Lead Zirconate Titanate (PZT)** powers nearly all modern piezoelectric devices (ultrasound, sensors, actuators). However, PZT contains >60% toxic lead, facing imminent bans under global regulations (RoHS).

Finding a replacement is an **optimization nightmare**:
1.  The search space for chemical solid solutions is effectively infinite.
2.  Complex stoichiometry (e.g., doping, substitution) makes traditional modeling difficult.
3.  Experimental synthesis is slow, expensive, and hazardous.

## 💡 The Solution

A comprehensive **Data-Driven Pipeline** that ingests raw chemical formulas and outputs validated property predictions.

1.  **Parse:** Custom regex algorithms break down complex, nested chemical strings into atomic feature vectors.
2.  **Learn:** Ensemble regression models learn non-linear relationships between atomic properties and macroscopic performance.
3.  **Deploy:** A production-grade web application serves these models to researchers globally.

---

## 🚀 Key Technical Features

### 1. 🧬 Advanced Stoichiometry Engineering
Unique to this project is a robust **Chemical Parsing Engine** capable of handling real-world, "messy" scientific notation.
*   **Nested Formula Support:** Recursively resolves complex solid solutions like `0.96(K0.48Na0.52)NbO3-0.04(Bi0.5Ag0.5)ZrO3`.
*   **Normalization:** Automatically balances stoichiometry and handles bracket variations (e.g., `[]` vs `()`) ensuring consistent feature generation regardless of user input style.
*   **Feature Vectors:** Maps cleaned formulas to 28+ atomic descriptors (electronegativity, ionic radius, valence electron concentration) based on domain knowledge.

### 2. 🧠 Adaptive Machine Learning Pipeline
The backend features a sophisticated, self-correcting ML engine (`ml_engine.py`):
*   **Auto-Tune vs. Expert Control:** 
    *   **Auto-Mode:** Automatically runs Grid Search (CV=5) across Random Forest, XGBoost, LightGBM, SVR, and Gradient Boosting to find the optimal architecture.
    *   **Manual Fine-Tuning:** Allows domain experts to override specific hyperparameters (e.g., `n_estimators`, `gamma`) for targeted experimentation.
*   **Ensemble Stacking:** Implements a `StackingRegressor` that combines weak learners to minimize variance and improve generalization on small datasets.
*   **Strict Parameter Sanitization:** A custom whitelist layer ensures model stability, preventing crashes when determining valid hyperparameters for different algorithms dynamically.

### 3. 📄 Automated Research Reporting
Bridging the gap between code and publication, the **PDF Reporting Engine** (`report_generator.py`) automates data storytelling:
*   **Dynamic Visualization:** Generates publication-ready vector graphics (Scatter plots with "Perfect Fit" lines, RMSE comparison bar charts).
*   **Contextual Insights:** Programmatically generates text summarizing "Best Performing Models" and "Model Certainty" based on training metrics.
*   **Smart Layouts:** Uses advanced `Flowable` logic to prevent charts and titles from splitting across pages, ensuring professional formatting.

### 4. ⚡ Real-Time Interactive UI
Built with **React 18** and **Vite**, the frontend prioritizes responsiveness and scientific accuracy:
*   **Formula Builder:** A specialized form component validating charge neutrality and chemical validity in real-time.
*   **Live Training Feedback:** Web-socket style polling provides granular progress updates ("Preprocessing", "Benchmarking", "Optimizing") to the user.
*   **Interactive Insights:** `Recharts`-powered graphs allow users to hover and inspect individual data points (e.g., outlier detection).

---

## 🛠️ Technology Stack

| Layer | Technology | Usage |
| :--- | :--- | :--- |
| **Frontend** | **React.js, Vite** | Component-based UI, fast HMR |
| | **Material UI (MUI)** | Enterprise-grade component library |
| | **Recharts** | Interactive data visualization |
| | **Framer Motion** | Physics-based animations |
| **Backend** | **FastAPI** | High-performance, async Python web server |
| **ML / Data** | **scikit-learn** | Pipeline construction, preprocessing, SVR, RF |
| | **XGBoost / LightGBM** | Gradient boosting implementations |
| | **pandas / numpy** | Vectorized data manipulation |
| | **chemparse** | Stoichiometry parsing basis |
| **Reporting** | **ReportLab** | Programmatic PDF generation |
| | **Matplotlib** | Static scientific plotting |
| **DevOps** | **Docker** | Containerization of full stack |

---

## 💻 Local Development Setup

### 1. Prerequisites
*   [Docker Desktop](https://www.docker.com/products/docker-desktop) (Recommended)
*   OR Python 3.9+ & Node.js 16+

### 2. Quick Start (Docker)
The easiest way to run the full application (Frontend + Backend + DB).
```bash
git clone https://github.com/tusaryan/Piezoelectrics-AI-Discovery-Lab.git
cd Piezoelectrics-AI-Discovery-Lab
docker-compose up --build
```
*   **App:** `http://localhost:3000`
*   **API Docs:** `http://localhost:8000/docs`

### 3. Manual Setup (Dev Mode)

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## 🔬 Scientific Validation Methodology

To ensure valid scientific outputs, the model follows a rigorous validation protocol:
1.  **Data Cleaning:** Removal of non-stoichiometric entries and duplicate formulas.
2.  **Stratified Split:** 80/20 Train-Test split to preserve distribution of target properties.
3.  **Metric Evaluation:** Models are scored on **$R^2$** (variance explained) and **RMSE** (average error features).
4.  **Target Scaling:** Implementation of `TransformedTargetRegressor` to handle non-normal distributions in target variables ($T_c$, $d_{33}$).

---

## 🔮 Future Roadmap

1.  **Inverse Design (Generative AI):** Implementing Variational Autoencoders (VAEs) to *generate* novel formulas with desired properties, rather than just predicting properties of known formulas.
2.  **Structure-Property Mapping:** Integrating Crystal Graph Convolutional Networks (CGCNN) to learn directly from crystal lattice files (CIFs).
3.  **Active Learning:** A "Human-in-the-loop" system that suggests the next best experiment to perform to maximally improve model confidence.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
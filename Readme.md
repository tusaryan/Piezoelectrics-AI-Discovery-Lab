# 🧪 AI-Assisted Discovery of New Lead-Free Piezoelectrics

![Project Status](https://img.shields.io/badge/Status-Active_Development-green)
![Tech Stack](https://img.shields.io/badge/Stack-FastAPI_React_Docker-blue)
![Domain](https://img.shields.io/badge/Domain-Materials_Informatics-purple)

## 📖 Overview

This project leverages **Materials Informatics** and **Machine Learning** to accelerate the discovery of high-performance, lead-free piezoelectric materials.

By training advanced regression models on chemical composition data, this application provides a "Virtual Laboratory" that can predict critical material properties—specifically the **Piezoelectric Coefficient ($d_{33}$)** and **Curie Temperature ($T_c$)**—**in seconds, replacing months of traditional trial-and-error experimentation.**

---

## 🌍 The Problem: The "Lead Dilemma"

For decades, **Lead Zirconate Titanate (PZT)** has been the industry standard for piezoelectric devices (sensors, actuators, ultrasound) due to its exceptional performance. However, PZT creates a significant global challenge:

* **Toxicity:** PZT contains over 60% lead by weight, a neurotoxin that poses severe health risks.
* **Environmental Impact:** E-waste containing PZT leaches lead into soil and groundwater, causing long-term contamination.
* **Regulatory Pressure:** Global directives like **RoHS** are restricting the use of hazardous substances, creating an urgent need for eco-friendly alternatives.

Finding a lead-free replacement is difficult because the search space of possible chemical combinations is vast. Traditional "cook and look" methods are too slow and expensive to explore this space effectively.

## 💡 The Solution

This project implements a **Data-Driven Workflow** to bypass traditional limitations:

1.  **Data Collection:** Aggregated data on lead-free ceramics (specifically KNN-based) from scientific literature.
2.  **Feature Engineering:** Developed a robust chemical parser to convert complex chemical formulas (including solid solutions and dopants) into numerical features.
3.  **Predictive Modeling:** Trained and compared multiple ensemble learning algorithms (XGBoost, Random Forest, LightGBM) to accurately predict material properties.
4.  **Web Application:** Wrapped the ML engine in a modern Full-Stack application, allowing researchers to easily test new compositions and retrain models with new data.

---

## 🚀 Features

### 1. 🧪 Interactive Property Prediction
* **Smart Parsing:** Handles complex stoichiometry, solid solutions (e.g., `0.96(KNN)-0.04(BNT)`), and dopants.
* **Dual-Target Prediction:** Simultaneously predicts:
    * **$d_{33}$:** Piezoelectric charge coefficient (pC/N).
    * **$T_c$:** Curie Temperature (°C).
* **Formula Builder:** A GUI-based tool to construct valid chemical formulas without typing errors.

### 2. 📊 Automated Model Training & Fine-Tuning
* **Dynamic Retraining:** Upload new datasets (`.csv`) directly via the UI to retrain the models.
* **Hyperparameter Tuning:** Adjust `n_estimators`, `learning_rate`, and `max_depth` via simple sliders.
* **Real-time Visualization:** Instantly generates Research-Grade plots to validate the new model:
    * **Comparison Bar Charts:** Compares $R^2$ and RMSE across multiple algorithms.
    * **Regression Scatter Plots:** Visualizes "Predicted vs. Experimental" values with ideal fit lines.

### 3. 💾 Model Management
* **Production Deployment:** "One-click" deployment of the best-performing model to the production environment.
* **Consistency:** Ensures the exact parsing logic used during training is applied during prediction.

---

## 🛠️ Technology Stack

### **Frontend**
* **React.js:** For a responsive, interactive user interface.
* **Vite:** Next-generation frontend tooling for fast builds and hot module replacement.
* **Material UI (MUI):** Comprehensive React UI component library for a polished look.
* **Recharts:** Composable charting library for visualizing model insights.
* **Framer Motion:** Production-ready motion library for React animations.
* **React Router:** Declarative routing for single-page applications.
* **Axios:** For seamless API communication.

### **Backend**
* **FastAPI:** High-performance Python web framework for serving ML models.
* **Machine Learning:** `scikit-learn`, `XGBoost`, `LightGBM` for regression tasks.
* **Data Processing:** `pandas`, `numpy`, and a custom `chemparse` wrapper for stoichiometry.
* **Visualization:** `matplotlib` and `seaborn` for server-side graph generation.

### **Infrastructure**
* **Docker & Docker Compose:** Containerizes both services for easy, consistent deployment across any environment.

---

## 💻 Local Development Setup

Follow these steps to run the application on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:

*   **[Git](https://git-scm.com/downloads):** For cloning the repository.
*   **[Docker Desktop](https://www.docker.com/products/docker-desktop):** (Recommended) For running the application in containers.
*   **[Python 3.9+](https://www.python.org/downloads/):** (For local backend development)
*   **[Node.js v16+](https://nodejs.org/):** (For local frontend development)

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/tusaryan/Piezoelectrics-AI-Discovery-Lab.git
    cd Piezoelectrics-AI-Discovery-Lab
    ```

### Running Locally (Without Docker)

If you prefer to run the application directly on your machine for debugging, follow the instructions for your operating system.

#### 1. Backend Setup (FastAPI)

**Prerequisites:** Python 3.9+ installed.

**macOS / Linux:**
1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Start the server:
    ```bash
    uvicorn main:app --reload --port 8000
    ```

**Windows:**
1.  Open PowerShell or Command Prompt and navigate to the backend directory:
    ```powershell
    cd backend
    ```
2.  Create and activate a virtual environment:
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  Install dependencies:
    ```powershell
    pip install -r requirements.txt
    ```
4.  Start the server:
    ```powershell
    uvicorn main:app --reload --port 8000
    ```

*The API will be available at `http://localhost:8000`.*

#### 2. Frontend Setup (React)

**Prerequisites:** Node.js (v16+) and npm installed.

**All Platforms (macOS, Windows, Linux):**
1.  Open a new terminal window and navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
    *The app will be available at `http://localhost:5173` (or the port shown in the terminal).*

### Docker Setup (Recommended for Production)
1.  **Run with Docker Compose**
    This single command builds both the Frontend and Backend containers and sets up the network.
    ```bash
    docker-compose up --build
    ```

2.  **Access the Application**
    *   **Web App:** Open [http://localhost:3000](http://localhost:3000) in your browser.
    *   **API Documentation:** Open [http://localhost:8000/docs](http://localhost:8000/docs) to see the Swagger UI.

### New Features
*   **Interactive Formula Builder:** Construct complex formulas with a user-friendly UI.
*   **Model Insights:** Visualize and compare model performance (R², RMSE) directly in the app.
*   **Advanced Retraining:** Upload new datasets and fine-tune models with "Auto-Tune" or manual parameter selection.


---

## 🔬 Scientific Validation

To ensure reliability, the model evaluates multiple algorithms before selection. This mirrors the methodology found in recent literature:

* **Algorithms Tested:** Random Forest, XGBoost, LightGBM, Gradient Boosting, SVR.
* **Metrics:** $R^2$ (Coefficient of Determination) and RMSE (Root Mean Square Error).
* **Validation:** Uses an 80/20 Train-Test split to prevent overfitting.

*Example Visualization generated by the app:*
> The system automatically plots predicted values against experimental values to visually confirm the model's accuracy on unseen data.

---

## 🔮 Future Roadmap

This project lays the groundwork for a complete autonomous discovery system. Future expansions include:

1.  **Inverse Design (Genetic Algorithms):**
    * Instead of just predicting properties for *known* formulas, use a Genetic Algorithm (GA) to *generate* new, hypothetical compositions and evolve them to maximize $d_{33}$ and $T_c$.
2.  **Deep Learning Integration:**
    * Implement Graph Neural Networks (GNNs) to better capture crystal structure relationships.
3.  **Microstructure Analysis:**
    * Integrate Computer Vision (CNNs) to analyze SEM (Scanning Electron Microscope) images and correlate microstructure with piezoelectric performance.
4.  **Expanded Property Prediction:**
    * Add models for Dielectric Loss ($\tan \delta$), Planar Coupling ($k_p$), and Density ($\rho$).

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

* Based on research methodologies for KNN-based ceramics.
* Inspired by recent advancements in ML-assisted materials discovery.
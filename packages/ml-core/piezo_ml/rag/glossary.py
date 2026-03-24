"""
Built-in materials science glossary for the Piezo.AI RAG knowledge base.
50 key terms auto-indexed on first startup.
"""

PIEZO_GLOSSARY: dict[str, str] = {
    "Piezoelectricity": "The ability of certain materials to generate an electric charge in response to applied mechanical stress, and vice versa.",
    "d33": "The piezoelectric charge coefficient measured along the poling direction (pC/N). Higher d33 means stronger piezoelectric response.",
    "Curie Temperature (Tc)": "The temperature above which a piezoelectric material loses its spontaneous polarization and piezoelectric properties.",
    "Perovskite": "ABO3 crystal structure common to most high-performance piezoelectric ceramics. A-site and B-site cations determine properties.",
    "KNN": "Potassium Sodium Niobate (K,Na)NbO3 — the most promising lead-free piezoelectric system, replacing the toxic PZT.",
    "PZT": "Lead Zirconate Titanate Pb(Zr,Ti)O3 — the industry standard piezoelectric but contains toxic lead (~60 wt%).",
    "MPB": "Morphotropic Phase Boundary — a composition boundary between two crystallographic phases where piezoelectric properties peak.",
    "PPT": "Polymorphic Phase Transition — temperature-induced phase boundary that can enhance piezoelectric response near room temperature.",
    "PVDF": "Polyvinylidene Fluoride — a piezoelectric polymer used in flexible sensors and composites. Much lower d33 than ceramics.",
    "Poling": "The process of applying a strong electric field to align ferroelectric domains, activating piezoelectric response.",
    "Ferroelectric": "A material with spontaneous electric polarization that can be reversed by an external electric field.",
    "Sintering": "High-temperature heat treatment that densifies ceramic powder into a solid polycrystalline body.",
    "Dopant": "An element added in small amounts to modify material properties (e.g., Li, Ta, Sb in KNN).",
    "A-site": "The larger cation site in the ABO3 perovskite structure, typically occupied by K, Na, Li, Bi, Ba, Ca, Sr.",
    "B-site": "The smaller cation site in the ABO3 perovskite structure, typically occupied by Nb, Ta, Sb, Ti, Zr, Hf.",
    "Goldschmidt Tolerance Factor": "t = (rA + rO) / √2(rB + rO). Values 0.9-1.0 favor perovskite stability. Below 0.9 → orthorhombic distortion.",
    "Octahedral Factor": "Ratio rB/rO. Predicts stability of the BO6 octahedra in the perovskite structure.",
    "Dielectric Constant (εr)": "Measure of a material's ability to store electrical energy. Higher εr often correlates with higher d33.",
    "Dielectric Loss (tan δ)": "Energy dissipated as heat during polarization cycling. Lower is better for sensor applications.",
    "Mechanical Quality Factor (Qm)": "Ratio of stored to dissipated energy during vibration. High Qm needed for resonant transducers.",
    "Planar Coupling (kp)": "Electromechanical coupling coefficient for radial mode vibration. Measures conversion efficiency.",
    "Remnant Polarization (Pr)": "The polarization remaining after the external electric field is removed. Higher Pr → better piezoelectric.",
    "Coercive Field (Ec)": "The electric field required to reduce polarization to zero. Lower Ec → easier poling.",
    "SHAP": "SHapley Additive exPlanations — a game-theoretic approach to explain individual ML predictions by assigning each feature a contribution value.",
    "XGBoost": "Extreme Gradient Boosting — an efficient gradient boosting algorithm. Primary model for d33 prediction in Piezo.AI.",
    "LightGBM": "Light Gradient Boosting Machine — a fast gradient boosting framework. Primary model for Tc prediction.",
    "Random Forest": "An ensemble of decision trees using bagging. Provides uncertainty estimates via tree variance.",
    "Gaussian Process Regression (GPR)": "A Bayesian ML method that provides calibrated uncertainty estimates. Ideal for active learning with small datasets.",
    "Optuna": "A Bayesian hyperparameter optimization framework using Tree-structured Parzen Estimator (TPE) sampling.",
    "Active Learning": "An ML strategy where the model selects the most informative samples for labeling, reducing experimental cost.",
    "Upper Confidence Bound (UCB)": "An acquisition function for active learning that balances exploitation (high prediction) and exploration (high uncertainty).",
    "Expected Improvement (EI)": "An acquisition function that selects candidates likely to improve upon the current best observation.",
    "NSGA-II": "Non-dominated Sorting Genetic Algorithm II — a multi-objective optimization algorithm that finds Pareto-optimal solutions.",
    "Pareto Front": "The set of solutions where no objective can be improved without worsening another. Used for d33-Tc-hardness tradeoffs.",
    "Symbolic Regression": "An ML technique that discovers explicit mathematical equations from data, unlike black-box models.",
    "PySR": "A Python/Julia symbolic regression library that evolves algebraic expressions to fit data.",
    "Transfer Learning": "Using a model pre-trained on a large related dataset to improve performance on a smaller target dataset.",
    "GNN": "Graph Neural Network — processes data as graph structures. CHGNet/ALIGNN use GNNs for materials property prediction.",
    "CHGNet": "Crystal Hamiltonian Graph Neural Network — a pretrained universal potential for materials science.",
    "Feature Engineering": "The process of creating input features from raw data. In Piezo.AI: elemental fractions + physics descriptors.",
    "Cross-Validation": "A technique for evaluating model generalization by training on subsets and testing on held-out data. K-fold is standard.",
    "Overfitting": "When a model memorizes training data instead of learning patterns. Detected by large gap between train and test R².",
    "Regularization": "Techniques to prevent overfitting: L1 (sparsity), L2 (smoothness), dropout (neural networks).",
    "Stacking Ensemble": "Combining multiple diverse models by training a meta-learner on their out-of-fold predictions.",
    "Vickers Hardness (HV)": "A measure of material hardness using a diamond pyramid indenter. Important for practical durability.",
    "Mohs Hardness": "A qualitative hardness scale from 1 (talc) to 10 (diamond), referenced against standard minerals.",
    "Electrospinning": "A fabrication method for creating PVDF nanofiber composites with enhanced β-phase content.",
    "Beta Phase (β-phase)": "The polar crystalline phase of PVDF responsible for its piezoelectric properties.",
    "Composite": "A material made from two or more constituents, e.g., KNN ceramic filler in PVDF polymer matrix.",
    "RAG": "Retrieval-Augmented Generation — combining LLM generation with retrieved context from a knowledge base for more accurate answers.",
}


def ensure_glossary_indexed(kb) -> int:
    """Index the glossary into the knowledge base if not already done."""
    stats = kb.get_stats()
    glossary_count = stats.get("by_type", {}).get("glossary", 0)
    if glossary_count >= len(PIEZO_GLOSSARY):
        return glossary_count  # Already indexed
    return kb.index_glossary(PIEZO_GLOSSARY)

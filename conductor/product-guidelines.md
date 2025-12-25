# Product Guidelines: Insider Threat Detection System

## 1. Guiding Principles
- **Modularity & Extensibility**: The system is designed with a modular architecture (preprocessing, feature engineering, models, ensemble) to facilitate the easy addition of new detection algorithms, data sources, or visualization components without disrupting the core pipeline.
- **Reproducibility**: All experiments, model training, and results must be fully reproducible. This is enforced through centralized configuration (seeds, hyperparameters), versioned datasets, and logged execution states.
- **Operational Efficiency**: The system prioritizes performance optimization, utilizing efficient libraries like Polars for data processing and "flash" model architectures to ensure viability on standard hardware (e.g., MacBook M4 with 16GB RAM) and potential future real-time applications.

## 2. Code Structure & Organization
- **Research-Oriented Engineering**: The codebase balances the flexibility required for academic research (e.g., easy ablation studies, hyperparameter tuning) with the structure needed for a functional prototype. It avoids over-engineering while maintaining clear separation of concerns to support the "Thesis" nature of the work.

## 3. Visual & Output Guidelines
- **Clear Visualizations**: Output high-quality, publication-ready plots (ROC curves, Confusion Matrices, t-SNE projections) to effectively communicate research findings and model performance.
- **Actionable Logs**: Generate comprehensive and structured logs to trace execution flow, debug issues, and monitor the system's operational health.
- **Standardized Metrics**: Employ a consistent set of evaluation metrics (AUC-ROC, Precision, Recall, F1) across all models to ensure fair and rigorous comparison.
- **Minimalist CLI**: Maintain a simple, focused command-line interface for executing pipeline stages, avoiding unnecessary complexity to streamline the user experience.

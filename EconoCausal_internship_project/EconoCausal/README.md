# EconoCausal: Dynamic Pricing via Double Machine Learning

EconoCausal is a beginner-friendly demonstration of estimating the causal effect of price on product demand using **Double Machine Learning (DML)**. It uses synthetic retail data, so the results demonstrate a method rather than prove a real-world pricing strategy.

## Research question
After adjusting for observed factors such as seasonality, marketing spend, competitor price, and baseline demand, how does price relate to units sold?

## Project structure
```text
EconoCausal/
├── app.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── dml.py
│   └── metrics.py
├── analysis/
│   └── eda.py
├── tests/
│   ├── test_data.py
│   └── test_dml.py
└── docs/
    ├── methodology.md
    ├── results_template.md
    └── limitations.md
```

## Setup (Python 3.10+ recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the analysis
```bash
python analysis/eda.py
python -m pytest
streamlit run app.py
```

The Streamlit app generates synthetic data, estimates the price effect, and displays a demand/revenue illustration for candidate prices.

## Method summary
DML uses cross-fitting to estimate nuisance relationships:
- `m(X) = E[Y | X]`, expected demand given observed controls.
- `g(X) = E[D | X]`, expected price given observed controls.
- Residualize demand and price, then estimate the effect using the residual-on-residual relationship.

The treatment is price; the outcome is units sold. The estimate is expressed as the estimated change in units sold for a one-unit increase in price, under the model and synthetic data assumptions.

## Reproducibility
The synthetic data generator uses a fixed random seed by default. The app also uses a fixed seed so results can be reproduced.

## Disclaimer
This is an educational portfolio project. Synthetic data cannot establish the true causal effect of pricing in a real business. Real deployment would require trustworthy data, careful identification assumptions, validation, and business constraints.

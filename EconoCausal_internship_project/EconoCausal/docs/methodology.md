# Methodology

## Question
What is the estimated effect of price (`price`) on demand (`units_sold`) after adjusting for observed controls?

## Variables
- Treatment: `price`
- Outcome: `units_sold`
- Controls: `seasonality`, `marketing_spend`, `competitor_price`, `baseline_demand`

## Double Machine Learning outline
1. Split observations into folds.
2. On each training fold, fit a model for expected demand given controls and another for expected price given controls.
3. Predict both quantities for the held-out fold.
4. Calculate residual demand and residual price.
5. Regress residual demand on residual price to obtain the partially linear effect estimate.

Cross-fitting helps reduce overfitting bias from the nuisance models. It does not automatically guarantee a valid causal estimate: assumptions about confounding, overlap, model quality, and data collection still matter.

## Synthetic data truth
The generator includes a price coefficient of -2.0 units sold per one price unit before clipping and random noise. Because observed units are clipped at zero and nuisance functions are estimated from finite samples, the estimated coefficient may not equal -2.0 exactly.

## Uncertainty
The code reports an approximate standard error for demonstration. It is not a fully validated inferential procedure and should not be used for formal claims without further statistical review.

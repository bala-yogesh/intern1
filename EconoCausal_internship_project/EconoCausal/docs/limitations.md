# Limitations and responsible interpretation

1. **Synthetic data:** all observations are generated for demonstration; no real customer or sales records are used.
2. **Causal assumptions:** DML cannot remove bias from unobserved confounders automatically. A causal interpretation requires defensible assumptions and appropriate data.
3. **Simple treatment effect:** the estimator assumes a partially linear relationship for the treatment effect; real price response may vary by product, customer, and time.
4. **Clipping:** simulated demand is clipped at zero, which means the exact linear effect is not constant at every possible value.
5. **Approximate uncertainty:** the standard-error calculation is educational and is not presented as a validated confidence interval.
6. **Pricing illustration:** the candidate-price table is a simple scenario calculation. It does not account for inventory, costs, competitors' reactions, fairness, or operational constraints.
7. **No production deployment:** do not use this demo to set actual prices without real data, domain review, validation, and monitoring.

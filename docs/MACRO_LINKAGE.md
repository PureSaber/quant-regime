# Frozen macro-change and cross-asset return research

`macro_linkage` in `evaluation.py` is only an availability/lineage join. The distinct
`quant_regime.linkage` API completes a research experiment: construct released macro
change features at each decision, train per-asset relations, freeze every estimate,
then compare forecasts of later interval returns with a matched baseline.
It uses the repository's existing `financial-data` extra and pinned QDK interface;
no new data provider or shared dependency contract is introduced.

## Inputs and timing

The return frame has timezone-aware `start`, `end`, `signal_at`, `return_known_at`
and one explicitly named simple-return column per asset. Intervals are ordered,
unique, contiguous and nonoverlapping, with `signal_at <= start < end` and
`return_known_at >= end`. Missing asset returns remain NaN. Returns must be finite
when present and greater than -1; supply complete total-return, currency and
instrument conventions appropriate to the research question. This module cannot
certify their provenance and performs no currency conversion or price interpolation.

Macro observations retain the existing QDK `financial.macro` contract, including
observation/vintage dates, released/available times, units, source and evidence IDs.
At each signal, QDK `macro_asof` selects only visible versions. The feature is the
absolute native-unit difference of the two latest visible **observation dates**.
It is not a difference between two downloads or an unobserved market consensus.
Current and previous values, observation dates, availability times and evidence IDs
are retained. Revisions may alter subsequent features only after publication; they
never rewrite an earlier signal. Null latest values, insufficient observations,
staleness and excessive observation gaps are explicit exclusions. Age/gap limits
are declared in days and frozen, not chosen from OOS results. Gaps within those
limits are still reported; a regular complete release calendar is not implied.

## Fit, freeze and compare

`fit_macro_relation(training, observations, assets=..., series=..., fit_at=...,
evidence_kind=...)` uses only intervals ending by `fit_at`, with return labels known
by then and all required macro features available at their own signal. For each
asset separately, mean/standard-deviation scaling, OLS intercept/slopes and a
constant-mean forecast baseline use **the identical eligible training rows**.
The model is frozen in immutable dataclasses. Coefficients are reported in native
macro units as well as stored standardized units. Too few observations, constant
features or rank deficiency leave the corresponding asset unavailable; if every
asset is unavailable, fitting is `blocked`. No feature or parameter search occurs.
Training missingness counts describe potentially overlapping reasons; `eligible_rows`
and `used_rows` report the actual fit sample separately.

`evaluate_macro_relation(model, testing, observations, evaluation_at=...)` requires
OOS intervals after training and signals at/after model availability. The evaluation
cutoff must reach all requested signal times. Future or missing return labels are
not exposed as observed returns or scored. Each prediction includes full interval
timing, its matched frozen-mean forecast, scoreability/exclusion reason and an
out-of-training-range feature flag. Available predictions with delayed outcomes
remain visible without fabricated scores.

Per asset, report candidate/predicted/scored/excluded counts, MSE, MAE, direction
accuracy and `1 - model_MSE / baseline_MSE`. Both models are scored on **exactly the
same OOS rows**; coverage is displayed so selective missingness is not hidden.
A zero baseline error makes relative R-squared undefined, not a pass. Negative
relative R-squared is retained. This benchmark is a forecasting benchmark, not a
buy-and-hold account. These are predictive associations, not causal macro effects,
statistical significance, net strategy returns or execution results. No public
schema, storage, Factors core or account ledger is changed.

## Reproducible result and independent controls

Run `python examples/macro_asset_linkage.py NEW_OUTPUT_DIRECTORY`. The synthetic
experiment fixes two orthogonal +/-1 macro-change series, 24 training intervals,
12 OOS intervals and equity/bond/commodity returns. Equity and bond preserve their
relations; the commodity growth loading deliberately reverses in OOS. The matched
constant-mean MSEs are respectively **0.0005, 0.00016 and 0.00036**. Frozen-model MSEs
are **0, 0 and 0.001296** up to floating-point roundoff, so relative R-squared is
**1, 1 and -2.6**. The deterioration is a reported research result, not screened out.

Balanced orthogonal features independently identify the specified native slopes;
baseline MSE equals the sum of squared true slopes. Commodity model error is
`0.036 * growth_change`, giving `0.036² = 0.001296`. The example asserts these
closed-form checks, saves complete inputs, frozen model, forecasts, coefficients,
comparisons and lineage, and hashes outputs in a manifest. A second evaluation of
the same frozen model injects a missing macro release, one missing bond return
and one late outcome: scored counts are equity **9**, bond **8**, commodity **9**
out of 12, with exclusions retained and baselines matched to those rows.

Tests additionally verify a future revision cannot alter the fitted model or prior
forecast, a later observable revision does change its subsequent feature, future
returns cannot change predictions, and timing/units/insufficient-sample failures do
not become success. The entire case is synthetic. Real macro/asset conclusions
still require archived vintage/publication evidence and sourced PIT return inputs.

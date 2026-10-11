# Frozen regime evaluation

`quant_regime.evaluation.fit_policy(training)` estimates a simple long/cash map from
training excess-return means. A state needs the declared minimum sample count. Its
scale is frozen at either zero or the declared long-only cap. This is one declared
candidate, not a search over thresholds. `evaluate_policy` compares it with a constant
same-cap policy on a strictly later contiguous interval sequence. An unseen state is
flat and reported. Both start flat and pay identical declared costs. Repeat this API
on predeclared, nonoverlapping train/test folds; do not tune the candidate on test results.
`compare_folds` accepts explicit `(train_start,test_start,test_end)` row boundaries,
rejects overlapping test intervals and returns separate fold comparisons. It does
not stitch independently reset sleeves into an account or hide fold liquidation costs.

Input columns are `start`, `end`, `signal_at` (timezone required), `regime`,
`asset_return`, `cash_return` (simple interval returns). `signal_at <= start < end`.
State labels must themselves be constructed causally; this API cannot prove upstream
labels did not use future data. Missing periods are rejected, not joined across gaps.
Costs are `bps/10000 * abs(target - drifted exposure)` on full asset turnover, including
entry. They debit the cash sleeve at interval end. Gross return is
`weight * asset_return + (1-weight) * cash_return`. After return and costs, drifted
exposure is `weight*(1+asset_return)/(1+net_return)`. This research convention permits
a small implicit cash debit at full exposure; it is not executable financing or a
certified account. Terminal holdings are marked, not silently liquidated.

Output contains period-level policy and benchmark returns, costs and weights;
state-conditioned performance, run lengths, switch counts and transitions. Conditional
compounding is descriptive and does not form an investable separate strategy. No
annualization is inferred from irregular timestamps. `macro_linkage` calls existing
QDK `macro_asof` through `macro_context` at each signal time, preserving evidence,
units, release timestamps and missing/stale reasons. It never overrides regime labels.
Linkage describes simultaneous information, not macro causation.

Run `python examples/evaluate_states.py` for a reproducible synthetic example.
Real historical-PIT evaluation additionally requires archived state inputs, vintage
release evidence, executable return intervals and evidenced cost assumptions. None is
manufactured by this module; synthetic acceptance is software evidence only.

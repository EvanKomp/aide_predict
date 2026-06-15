# Making UCB β mode-consistent for the delta model

**Problem.** The UCB exploration coefficient β currently multiplies a *different physical
quantity* depending on the aggregation mode. For single-reference modes (`nearest`,
`formaldehyde`) it scales pure model uncertainty; for multi-reference modes (`avg_all`,
`distance_weighted`) it additionally scales the *disagreement between reference substrates*.
This note derives the current math, isolates the culprit term, and proposes a fix that makes
β mean the same thing in every mode.

---

## 1. Setup / notation

For one target (mutation $m$, held-out substrate $s$), the model produces a prediction from
each candidate reference substrate $r$. Because it is a **delta model**, each per-reference
absolute prediction is

$$\mu_r = \underbrace{\delta_r}_{\text{predicted } \log fc - \log fc_{\text{ref}}}
        + \underbrace{c_r}_{\log fc_{\text{ref}}(m,r)}$$

with per-reference epistemic and aleatoric standard deviations $e_r$, $a_r$ (from
weight-posterior MC sampling and the predicted log-variance, respectively).

> $c_r$ is a **constant added to every MC sample** (`05_bnn2_common.py:366-367, 415-416`),
> so it shifts $\mu_r$ but does **not** enter $e_r$ or $a_r$. The per-pair uncertainty is the
> model's uncertainty about the *delta*, not about the reference value.

Aggregation weights $w_r$ (with $\sum_r w_r = 1$) depend on the mode
(`aggregate_pairwise_predictions:1576-1590`):

| Mode | $w_r$ | Effective $n_{\text{ref}}$ |
|---|---|---|
| `nearest` | one-hot on closest reference | 1 |
| `formaldehyde` | filtered to formaldehyde-as-ref | ~1 |
| `distance_weighted` | $\operatorname{softmax}(-d_r/\tau)$ | many (near-weighted) |
| `avg_all` ("mean") | $1/n_{\text{ref}}$ | all |

---

## 2. Current math

Aggregation via the **law of total variance** (`aggregate_pairwise_predictions:1592-1600`):

$$\bar\mu = \sum_r w_r \mu_r$$

$$
V_{\text{epi}}
= \underbrace{\sum_r w_r e_r^2}_{V_{\text{within}}}
+ \underbrace{\sum_r w_r(\mu_r-\bar\mu)^2}_{\text{var\_means (between)}},
\qquad
V_{\text{ale}} = \sum_r w_r a_r^2
$$

$$
\sigma_{\text{tot}}
= \sqrt{V_{\text{epi}} + V_{\text{ale}}}
= \sqrt{\,V_{\text{within}} + \text{var\_means} + V_{\text{ale}}\,}
$$

and the acquisition score used everywhere (β-sweep, recovery curves) is

$$\text{UCB} = \bar\mu + \beta\,\sigma_{\text{tot}}.$$

### The inconsistency

`var_means` is **structurally zero** when there is one reference (`nearest`, `formaldehyde`)
and **strictly positive** for `avg_all` / `distance_weighted`:

$$
\sigma_{\text{tot}}^{\text{single-ref}} = \sqrt{e^2 + a^2}
\quad\text{(pure model uncertainty)}
\qquad\text{vs}\qquad
\sigma_{\text{tot}}^{\text{multi-ref}} = \sqrt{V_{\text{within}} + \text{var\_means} + V_{\text{ale}}}.
$$

So β multiplies a different quantity in each mode — the asymmetry we want to remove.

---

## 3. What `var_means` actually measures — and why it pollutes ranking

$$\text{var\_means} = \sum_r w_r(\mu_r - \bar\mu)^2 = \operatorname{Var}_r\big(\delta_r + c_r\big).$$

If the predicted delta were reference-invariant ($\delta_r \approx \text{const}$), this collapses to

$$\text{var\_means} \approx \operatorname{Var}_r(c_r) = \operatorname{Var}_r\!\big(\log fc_{\text{ref}}(m,r)\big),$$

i.e. **purely the spread of the mutation's measured activity across the reference panel**. In
practice $\delta_r$ varies with $r$ too, but a large share of `var_means` is still that
cross-substrate activity spread.

**Consequence.** For `avg_all`, $\beta\,\sigma_{\text{tot}}$ up-ranks **mutations that are highly
variable across substrates** — a property of the mutation's promiscuity, *not* of how
confident or good the prediction is on the held-out target. This is why β helped the
single-reference modes but hurt `avg_all` in the sweep.

---

## 4. Proposed fix: acquisition σ = within-reference uncertainty only

Make β scale the **model's predictive uncertainty about each reference's estimate**, excluding
the between-reference disagreement:

$$
\boxed{\;\sigma_{\text{acq}}^2 = \sum_r w_r\big(e_r^2 + a_r^2\big) = \sigma_{\text{tot}}^2 - \text{var\_means}\;}
$$

$$\text{UCB} = \bar\mu + \beta\,\sigma_{\text{acq}}.$$

- **single-ref:** $\sigma_{\text{acq}} = \sqrt{e^2+a^2}$ — **identical to today** (var_means was
  already 0). No change to `nearest` / `formaldehyde`.
- **multi-ref:** $\sigma_{\text{acq}}$ is the weight-averaged per-reference model uncertainty,
  with the reference-spread term removed. β now means the same thing in every mode: *"bonus
  proportional to how uncertain the model is, not how much the reference panel disagrees."*

### Equivalent "score-then-aggregate" view

Scoring each reference and then aggregating,

$$\sum_r w_r\big(\mu_r + \beta\sigma_r\big) = \bar\mu + \beta\sum_r w_r\sigma_r,
\qquad \sigma_r = \sqrt{e_r^2 + a_r^2},$$

also removes var_means and reduces to $\sigma_r$ in single-ref. It uses the **L1** combination
$\sum_r w_r\sigma_r$; the boxed version uses the **L2/quadrature** $\sqrt{\sum_r w_r\sigma_r^2}$.
Both are valid; the quadrature form is preferred since it is literally "drop one term from the
law of total variance," keeping the variance algebra consistent.

---

## 5. Orthogonal choice: aleatoric in or out?

Classic UCB / active learning explores **reducible (epistemic)** uncertainty — there is no
value in exploring where noise is irreducible. A more principled variant drops aleatoric too:

$$\sigma_{\text{acq}}^2 = \sum_r w_r\, e_r^2 \qquad(\text{epistemic-only, also var\_means-free}).$$

This is **independent** of the var_means fix — either σ choice can be epistemic-only or
epistemic+aleatoric. Both make β mode-consistent; epistemic-only additionally makes β target
*learnable* uncertainty.

| Variant | $\sigma_{\text{acq}}^2$ | β scales | single-ref vs today |
|---|---|---|---|
| Current | $V_{\text{within}} + \text{var\_means} + V_{\text{ale}}$ | total predictive var (mode-dependent) | — |
| Proposed A (epi+ale) | $V_{\text{within}} + V_{\text{ale}}$ | model predictive uncertainty | identical |
| Proposed B (epi-only) | $V_{\text{within}}$ | reducible uncertainty | smaller by $a^2$ |

---

## 6. What does **not** change

`var_means` is a legitimate component of the **honest predictive variance** of the aggregated
estimator, so it must stay in the σ used for **calibration / NLPD / CRPS / sharpness**
(`compute_nlpd`, `compute_calibration`, `compute_crps_gaussian`, …). The proposal swaps the σ
used **only in the UCB ranking term**. We carry two quantities:

- `tot_std` $= \sigma_{\text{tot}}$ (full law-of-total-variance) — for reporting + all
  uncertainty diagnostics. **Unchanged.**
- `acq_std` $= \sigma_{\text{acq}}$ (var_means-free) — for the β-ranking only. **New.**

---

## 7. Implementation feasibility (no retraining)

Everything is reconstructable at **scoring time**. `pairwise_predictions.csv` already stores
per-(mutation, reference) `epi_std` and `ale_std`, so the scorer's re-aggregation
(`aggregate_and_null_per_fold` → `aggregate_pairwise_predictions`) can compute $V_{\text{within}}$,
`var_means`, and $V_{\text{ale}}$ directly and emit an extra `acq_std` column. Then:

- `compute_beta_sweep_for_mode` and `_acq_scores` rank by `y_pred + β·acq_std` instead of
  `y_pred + β·tot_std`.
- The headline `bnn_at_best_beta` block uses `acq_std` as well, for consistency.

**Wrinkle.** `aggregate_pairwise_predictions` lives in `05_bnn2_common.py` (shared with the
legacy pipeline). Expose the new column **additively** (legacy selects fixed columns, so it is
backward-compatible) — or compute `acq_std` in the scorer's wrapper to avoid touching the
shared file at all.

---

## 8. Recommendation

1. **Adopt the var_means-free acquisition σ** — the actual fix for "β means the same across
   modes," and a no-op for the single-reference modes already trusted.
2. **Make it epistemic-only** ($\sigma_{\text{acq}}^2 = \sum_r w_r e_r^2$) — cleanest UCB
   semantics; also removes a second cross-mode inconsistency (aleatoric magnitude can differ by
   mode).
3. Keep full `tot_std` for calibration / NLPD untouched.

**Expected effect:** `nearest` / `formaldehyde` curves essentially unchanged;
`avg_all` / `distance_weighted` β behavior becomes a genuine model-uncertainty bonus rather
than a cross-substrate-variance bonus — so β should help or be neutral there instead of
actively hurting.

**Open fork (determines implementation):** acquisition σ **epistemic-only** (recommended) vs
**epistemic + aleatoric** (matches today's single-ref σ magnitude exactly). Everything else
follows from this choice.

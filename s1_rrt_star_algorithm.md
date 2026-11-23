# S1-RRT*: System-1 Guided Rapidly-Exploring Random Tree Star

## (a) Key Idea

**Core mechanism.** Run standard RRT\* but draw each sample from the mixture

$$p(x) \;=\; \varepsilon \cdot \mathcal{U}(X_{\mathrm{free}}) \;+\; (1-\varepsilon)\cdot q(x), \qquad \varepsilon \in (0,1],$$

where $\mathcal{U}(X_{\mathrm{free}}) = 1/\mu(X_{\mathrm{free}})$ is the uniform density over free space and $q$ is *any* density supplied by a learned "System 1" model (neural network, VLM, inverse-RL policy, etc.). The model $q$ may be arbitrarily wrong — the $\varepsilon$-uniform component alone carries the theoretical guarantees. System 1 additionally provides an ordering heuristic for parent selection and rewiring, which is a pure computational optimisation that does not affect correctness.

**Why it works.** The Karaman–Frazzoli (2011) proof of asymptotic optimality for RRT\* rests on a *covering-ball argument*: every ball of radius $r_n$ along the optimal path must eventually contain a sample, and adjacent balls' samples must be connectable. This argument requires only that the sampling density is bounded below by some $p_{\min}>0$ on $X_{\mathrm{free}}$. Under our mixture, $p(x) \geq \varepsilon / \mu(X_{\mathrm{free}}) > 0$ everywhere in $X_{\mathrm{free}}$, so the argument goes through with the connection-radius constant $\gamma$ inflated by the factor $\varepsilon^{-1/d}$.

**What is not novel.** The mixture-with-uniform-fallback technique is a known approach to preserving sampling-based guarantees under biased distributions (it appears implicitly in analyses of Informed RRT\*, goal-biased RRT, and similar). The contribution here is:
1. A self-contained, explicit derivation of the adjusted constant $\gamma_\varepsilon$ and the exact requirements on $\varepsilon$ under which all guarantees survive.
2. A specification of how a *general-purpose learned model* (System 1) interfaces with RRT\* beyond sampling — via parent/rewire ordering — without affecting the proof.
3. An honest analysis of the finite-budget benefit, the cost of the guarantee, and the failure modes.

---

## (b) Algorithm

### Notation

| Symbol | Meaning |
|---|---|
| $X \subset \mathbb{R}^d$ | Compact configuration space |
| $X_{\mathrm{obs}}$ | Obstacle region (closed) |
| $X_{\mathrm{free}} = \mathrm{int}(X \setminus X_{\mathrm{obs}})$ | Open free space |
| $\mu(\cdot)$ | Lebesgue measure |
| $\zeta_d = \pi^{d/2}/\Gamma(d/2+1)$ | Volume of the unit $d$-ball |
| $c(\cdot,\cdot)$ | Edge cost (see Assumptions) |
| $\eta$ | Maximum steering step size |
| $\texttt{S1}$ | The System 1 oracle |

### Pseudocode

```
Algorithm S1-RRT*(x_init, x_goal, S1, ε, N)
────────────────────────────────────────────────
  V ← {x_init};  E ← ∅;  cost[x_init] ← 0

  for n = 1, 2, …, N do

    ── STEP 1: Mixture Sampling ──────────────────
    u ~ Uniform(0,1)
    if u < ε then                          // uniform fallback
        x_rand ← UniformSample(X_free)
    else                                   // System 1 proposal
        x_rand ← S1.Sample(V, E, x_goal)
        if x_rand ∉ X_free then continue   // reject invalid proposals

    ── STEP 2: Nearest + Steer ───────────────────
    x_nearest ← Nearest(V, x_rand)
    x_new     ← Steer(x_nearest, x_rand, η)
    if ¬CollisionFree(x_nearest, x_new) then continue

    ── STEP 3: Connection Radius ─────────────────
    r_n ← γ_ε · (log(|V|) / |V|)^{1/d}   // γ_ε defined in §(c)
    X_near ← {v ∈ V : ‖v − x_new‖ ≤ min(r_n, η)}

    ── STEP 4: Best Parent (S1-ordered) ──────────
    X_near_sorted ← S1.OrderByPriority(X_near, x_new)
    x_min ← x_nearest
    c_min ← cost[x_nearest] + c(x_nearest, x_new)

    for x_near in X_near_sorted do
        if CollisionFree(x_near, x_new) then
            c_try ← cost[x_near] + c(x_near, x_new)
            if c_try < c_min then
                x_min ← x_near;  c_min ← c_try

    V ← V ∪ {x_new}
    E ← E ∪ {(x_min, x_new)}
    cost[x_new] ← c_min

    ── STEP 5: Rewire (S1-ordered) ───────────────
    for x_near in X_near_sorted do       // same ordering
        c_try ← cost[x_new] + c(x_new, x_near)
        if c_try < cost[x_near] then
            if CollisionFree(x_new, x_near) then
                parent[x_near] ← x_new
                cost[x_near]   ← c_try
                PropagateUpdate(x_near)   // cascade cost updates

  return (V, E)
```

### Role of System 1

| S1 output | How it enters | Affects guarantees? |
|---|---|---|
| `S1.Sample(V, E, x_goal)` → candidate $x \in \mathbb{R}^d$ | Mixed with $\varepsilon$-uniform in Step 1 | **No** — density bounded below by $\varepsilon/\mu(X_{\mathrm{free}})$ regardless of $q$ |
| `S1.OrderByPriority(X_near, x_new)` → permutation of $X_{\text{near}}$ | Determines iteration order in Steps 4–5 | **No** — all neighbours are still visited; only the order changes |

> [!IMPORTANT]
> System 1 never *removes* candidates from the neighbour set or *blocks* the uniform sample. It only *proposes* and *reorders*. This is the structural reason the guarantee is $q$-independent.

---

## (c) Proof Sketch with Assumptions

### Assumptions

**(A1) Space.** $X = [0,1]^d$; $X_{\mathrm{free}}$ is a non-empty open subset with $\mu(X_{\mathrm{free}}) > 0$.

**(A2) Clearance.** The optimal path $\sigma^*:[0,1]\to X_{\mathrm{free}}$ has *$\delta$-clearance*: for every $t$, the open ball $B(\sigma^*(t), \delta) \subset X_{\mathrm{free}}$ for some $\delta > 0$.

**(A3) Cost.** The cost functional is path length: $c(\sigma) = \int_0^1 \|\dot\sigma(t)\| \, dt$. The edge cost satisfies $c(x,y) = \|x-y\|$ (Euclidean). Extension to costs satisfying the Lipschitz-continuity and triangle-inequality conditions of Karaman–Frazzoli (2011, §III-B) is straightforward.

**(A4) System 1 density.** $q$ is any Borel probability density on $X_{\mathrm{free}}$ (or any distribution from which we can generate i.i.d. samples; it may place all mass on a single point, concentrate on the wrong homotopy class, etc.).

**(A5) Mixture parameter.** $\varepsilon \in (0,1]$ is a fixed constant (not decaying with $n$).

**(A6) Connection radius.** We use

$$r_n \;=\; \gamma_\varepsilon \cdot \left(\frac{\log n}{n}\right)^{1/d}$$

with

$$\boxed{\;\gamma_\varepsilon^{\,d} \;>\; 2^d\!\left(1+\tfrac{1}{d}\right) \cdot \frac{\mu(X_{\mathrm{free}})}{\varepsilon \cdot \zeta_d}\;}$$

Equivalently, defining the standard RRT\* critical constant

$$\gamma_*^{\,d} = 2^d\!\left(1+\tfrac{1}{d}\right) \cdot \frac{\mu(X_{\mathrm{free}})}{\zeta_d}$$

we have

$$\gamma_\varepsilon \;=\; \gamma_* \cdot \varepsilon^{-1/d}.$$

> [!NOTE]
> The only change from standard RRT\* is the factor $\varepsilon^{-1/d}$ in the connection radius. For $\varepsilon = 1$ (pure uniform), we recover standard RRT\*.

---

### Theorem 1 (Probabilistic Completeness)

**Statement.** Under (A1)–(A5), if a feasible path from $x_{\mathrm{init}}$ to $x_{\mathrm{goal}}$ exists in $X_{\mathrm{free}}$, then

$$\Pr\bigl[\text{S1-RRT* finds a feasible path within } n \text{ iterations}\bigr] \;\xrightarrow{n\to\infty}\; 1.$$

**Proof sketch.**

The argument is identical to that for standard RRT (LaValle & Kuffner, 2001). The key requirement is that every open subset $U \subseteq X_{\mathrm{free}}$ with $\mu(U) > 0$ is sampled infinitely often a.s.

Under the mixture, each sample falls in $U$ with probability

$$\Pr[x_n \in U] \;\geq\; \varepsilon \cdot \frac{\mu(U)}{\mu(X_{\mathrm{free}})} \;>\; 0.$$

Samples are i.i.d., so by the second Borel–Cantelli lemma (events are independent, probabilities sum to $+\infty$), every such $U$ is hit infinitely often a.s. The remainder of the completeness proof — Voronoi bias ensures the tree expands toward uncovered regions — is unchanged. $\square$

---

### Theorem 2 (Asymptotic Optimality)

**Statement.** Under (A1)–(A6), let $c_n$ be the cost of the best path in the S1-RRT\* tree after $n$ samples. Then

$$c_n \;\xrightarrow{n\to\infty}\; c^* \quad \text{almost surely,}$$

where $c^* = c(\sigma^*)$ is the optimal cost.

**Proof sketch.** We follow the covering-ball strategy of Karaman–Frazzoli (2011, Theorem 38), tracking where the factor $\varepsilon$ enters.

**Step 1 — Path discretization.**
Cover $\sigma^*$ by $m$ overlapping balls $B_1, \ldots, B_m$ of radius $\delta_n = r_n / 2$, centered at equally spaced points along $\sigma^*$. The number of balls is

$$m \;=\; \bigl\lceil L^* / \delta_n \bigr\rceil \;=\; O\!\left(\frac{L^*}{\gamma_\varepsilon} \cdot \left(\frac{n}{\log n}\right)^{1/d}\right),$$

where $L^* = c^*$ is the path length. By assumption (A2), for $n$ large enough that $\delta_n < \delta$, every $B_i \subset X_{\mathrm{free}}$.

**Step 2 — Ball coverage probability.**
The probability that ball $B_i$ receives no sample from the mixture after $n$ i.i.d. draws is

$$\Pr[\text{no sample in } B_i] \;=\; \prod_{j=1}^{n}\left(1 - \int_{B_i} p(x)\,dx\right) \;\leq\; \left(1 - \varepsilon \cdot \frac{\zeta_d \, \delta_n^d}{\mu(X_{\mathrm{free}})}\right)^n.$$

Substituting $\delta_n = (\gamma_\varepsilon / 2)(\log n / n)^{1/d}$:

$$\leq\; \exp\!\left(-\,n \cdot \varepsilon \cdot \frac{\zeta_d \, \gamma_\varepsilon^d}{2^d \, \mu(X_{\mathrm{free}})} \cdot \frac{\log n}{n}\right) \;=\; n^{-\alpha},$$

where

$$\alpha \;=\; \frac{\varepsilon \,\zeta_d \,\gamma_\varepsilon^d}{2^d \,\mu(X_{\mathrm{free}})}.$$

By assumption (A6), $\alpha > 1 + 1/d$.

**Step 3 — Union bound.**
The probability that *any* ball is empty is

$$\Pr[\exists\, i : B_i \text{ empty}] \;\leq\; m \cdot n^{-\alpha} \;=\; O\!\left(n^{1/d - \alpha} \cdot (\log n)^{-1/d}\right).$$

Since $\alpha > 1 + 1/d$, the exponent $1/d - \alpha < -1$, so this probability is summable in $n$. By the first Borel–Cantelli lemma, eventually all balls are occupied a.s.

**Step 4 — Connectivity.**
When both $B_i$ and $B_{i+1}$ contain samples $v_i, v_{i+1}$, their distance satisfies $\|v_i - v_{i+1}\| \leq 2\delta_n = r_n$. Therefore $v_{i+1}$ is in the connection neighbourhood of $v_i$, and (since $B_i \cup B_{i+1} \subset X_{\mathrm{free}}$ by clearance) the straight-line edge is collision-free. The RRT\* rewiring step ensures that $v_{i+1}$ is connected to $v_i$ (or a better parent) in the tree.

**Step 5 — Cost convergence.**
The cost of the path through the samples $v_1, \ldots, v_m$ satisfies

$$\sum_{i=1}^{m-1} \|v_i - v_{i+1}\| \;\leq\; \sum_{i=1}^{m-1} 2\delta_n \;\leq\; L^* + 2\delta_n \;\xrightarrow{n\to\infty}\; L^*,$$

since $\delta_n \to 0$ and the sample within each ball can be chosen arbitrarily close to the path. (The formal argument uses the result that the "good event" — all balls occupied, cost within $\epsilon$ of optimal — occurs for all but finitely many $n$ a.s., which is exactly what Steps 2–3 establish.) $\square$

---

### Corollary (k-Nearest Variant)

> [!NOTE]
> **Conjecture (not fully proven here).** The k-nearest variant with $k_n = k \cdot \log n$ and $k > 2^d \cdot e \cdot (1+1/d)$ retains asymptotic optimality under mixture sampling without any $\varepsilon$-dependent adjustment to $k$.
>
> **Reasoning.** In regions where the density is close to the minimum $\varepsilon / \mu(X_{\mathrm{free}})$, the k-nearest radius is approximately $(k \log n \cdot \mu(X_{\mathrm{free}}) / (n \varepsilon \zeta_d))^{1/d} = (k\mu(X_{\mathrm{free}})/(\varepsilon \zeta_d))^{1/d} \cdot (\log n / n)^{1/d}$, which is $\varepsilon^{-1/d}$ times the uniform-case radius. This automatically compensates for the sparser local density, playing the same role as $\gamma_\varepsilon$ in the radius-based variant. The compensating inflation is emergent rather than tuned. A full proof would require establishing that the k-nearest graph under non-uniform sampling satisfies the connectivity conditions of the covering-ball argument at every point on $\sigma^*$, which involves local-density-dependent bounds on the k-nearest radius that track through the Karaman–Frazzoli argument. We mark this as a conjecture pending a complete proof.

---

## (d) Complexity and Limitations

### Per-Iteration Complexity

| Operation | Standard RRT\* | S1-RRT\* |
|---|---|---|
| **Sample generation** | $O(1)$ | $O(T_{\mathrm{S1}})$ where $T_{\mathrm{S1}}$ is the S1 inference cost (e.g. a forward pass) |
| **Nearest neighbour** | $O(\log n)$ with kd-tree | Same |
| **Neighbour query** | $O(\log n + |X_{\text{near}}|)$ | Same, but $|X_{\text{near}}|$ may be larger (see below) |
| **Collision checks** (parent + rewire) | $O(|X_{\text{near}}|)$ | Same in worst case; $O(1)$ in best case with S1 ordering and lazy evaluation |
| **Connection radius** | $\gamma_* (\log n / n)^{1/d}$ | $\gamma_\varepsilon (\log n / n)^{1/d}$, larger by factor $\varepsilon^{-1/d}$ |

**Expected neighbour count.** At a point $x$ with mixture density $p(x) = \varepsilon/\mu(X_{\mathrm{free}}) + (1-\varepsilon)q(x)$:

$$\mathbb{E}[|X_{\text{near}}|] \;=\; n \cdot p(x) \cdot \zeta_d \cdot r_n^d \;\approx\; \bigl(\varepsilon + (1-\varepsilon)\,q(x)\,\mu(X_{\mathrm{free}})\bigr) \cdot \frac{\zeta_d \, \gamma_\varepsilon^d \, \log n}{\mu(X_{\mathrm{free}})}.$$

- In low-density regions ($q(x) \approx 0$): $\approx \varepsilon \cdot \gamma_\varepsilon^d \cdot \zeta_d \cdot \log(n)/\mu(X_{\mathrm{free}}) = \gamma_*^d \cdot \zeta_d \cdot \log(n) / \mu(X_{\mathrm{free}}) = O(\log n)$, matching standard RRT\*.
- In high-density regions ($q(x) \gg 1/\mu(X_{\mathrm{free}})$): can be $\Theta(q(x) \cdot \mu(X_{\mathrm{free}}) \cdot \log n / \varepsilon)$, potentially much larger.

**Mitigation.** Cap $|X_{\text{near}}|$ at $k_{\max} \cdot \log n$ for a chosen constant $k_{\max}$, selecting the $k_{\max} \cdot \log n$ nearest neighbours. This is equivalent to switching to the k-nearest variant in dense regions and preserves guarantees (by the conjecture in §(c), or by falling back to the radius guarantee in sparse regions).

---

### Finite-Budget Benefit

**When System 1 is good.** If $q$ concentrates mass near $\sigma^*$, the tree is rapidly populated along the optimal path. The first feasible solution and early cost reductions arrive much sooner than under uniform sampling.

Quantitatively: suppose $q$ places fraction $\beta$ of its mass within distance $\delta$ of $\sigma^*$. After $n$ iterations, the expected number of samples near $\sigma^*$ is

$$\mathbb{E}[n_{\text{near}}] \;\approx\; n\bigl(\varepsilon \cdot \delta^d \zeta_d L^* / \mu(X_{\mathrm{free}}) + (1-\varepsilon)\beta\bigr).$$

Under pure uniform sampling, $n_{\text{near}} \approx n \cdot \delta^d \zeta_d L^* / \mu(X_{\mathrm{free}})$, which can be orders of magnitude smaller in high $d$ when $\delta$ is small and $\beta$ is large. The finite-budget improvement ratio is approximately

$$\frac{n_{\text{near}}^{\text{S1-RRT*}}}{n_{\text{near}}^{\text{RRT*}}} \;\approx\; 1 + \frac{(1-\varepsilon)\beta \, \mu(X_{\mathrm{free}})}{\varepsilon \,\delta^d \, \zeta_d \, L^*}.$$

This ratio grows as $\delta^{-d}$: the narrower the optimal corridor, the greater the advantage of a good heuristic.

**When System 1 is bad.** The $(1-\varepsilon)$ fraction of samples from $q$ is effectively wasted for coverage purposes. The algorithm behaves as RRT\* with only $\varepsilon \cdot n$ effective uniform samples. The convergence rate to $c^*$ is therefore slower by approximately factor $1/\varepsilon$ compared to pure RRT\*. Asymptotic optimality is unaffected; only the constant in the convergence rate changes.

> [!IMPORTANT]
> **The cost of the guarantee is a $1/\varepsilon$ slowdown in the convergence rate (not a change in rate order), plus a $\varepsilon^{-1/d}$ increase in connection radius (increasing per-iteration work in high-density regions).** For $\varepsilon = 0.1$ in $d=3$, $\varepsilon^{-1/d} \approx 2.15$, which is modest.

---

### Failure Modes

#### 1. Homotopy Lock-In

**Description.** If $q$ strongly favours one homotopy class (e.g. a narrow passage the System 1 model has never seen), the tree becomes dense in that class. Samples from the $\varepsilon$-uniform component that land in the correct class must still connect to the tree via the narrow passage, which requires multiple uniform samples in the passage simultaneously.

**Severity.** This is a *finite-budget* problem, not an asymptotic one. The ε-uniform component guarantees that the correct homotopy class is eventually explored (by completeness). However, the time to discover the alternative class scales exponentially in the passage width (same as standard RRT\*).

**Mitigation.** Combine with passage-detection heuristics (e.g. Gaussian bridge sampling, retraction-based sampling) as an additional mixture component. This adds another term to $p(x)$ without invalidating the lower bound.

#### 2. Rewiring Starvation

**Description.** In regions where $q$ has very high density, the tree accumulates many nodes. Each new node in such a region triggers rewiring over $\Theta(q_{\max} \cdot \log n)$ neighbours, consuming collision-check budget without proportional cost improvement (the region is already well-explored).

**Severity.** This is a *computational efficiency* problem. It does not affect correctness or asymptotic optimality, but degrades wall-clock performance.

**Mitigation.**
- *Lazy collision checking*: skip collision checks for rewiring candidates whose cost improvement is below a threshold.
- *Neighbour count cap*: limit $|X_{\text{near}}|$ to $k_{\max} \cdot \log n$.
- *S1 ordering*: by ordering rewire candidates by decreasing expected cost improvement (from S1's value function), the most impactful rewires happen first, and the loop can terminate early.

#### 3. Radius Mis-Scaling Under Non-Uniform Density

**Description.** The adjusted radius $r_n = \gamma_\varepsilon (\log n / n)^{1/d}$ is globally uniform but the density $p(x)$ is spatially non-uniform. In dense regions, $r_n$ may be unnecessarily large (creating too many connections); in sparse regions, it is exactly sufficient.

**Severity.** Moderate. The global radius is a worst-case bound designed for the sparsest regions. In dense regions it creates superfluous connections that increase computational cost without improving solution quality.

**Mitigation.** Use the k-nearest variant, which implicitly adapts the effective radius to the local density. The conjectured guarantee (see §(c) Corollary) suggests this works without $\varepsilon$-dependent tuning of $k$. Alternatively, use a spatially-varying radius $r_n(x) = \gamma_\varepsilon \cdot (\log n / (n \cdot \hat{p}_n(x) \cdot \mu(X_{\mathrm{free}})))^{1/d}$ where $\hat{p}_n(x)$ is a local density estimate. However, the theoretical analysis of spatially-varying radii under non-uniform sampling is nontrivial and we leave it as an open problem.

#### 4. System 1 Inference Latency

**Description.** If $T_{\mathrm{S1}}$ (the cost of one S1 forward pass) is comparable to or exceeds the cost of a collision check, the per-iteration overhead of querying System 1 may negate the benefit of fewer iterations to convergence.

**Mitigation.** Batch S1 queries (generate $B$ proposals at once, amortizing the inference cost), or use a lightweight distilled model. The mixture parameter $\varepsilon$ also acts as a duty-cycle control: only $(1-\varepsilon)$ fraction of iterations pay the S1 cost.

---

### Summary Table of Quantities

| Quantity | Standard RRT\* | S1-RRT\* | Ratio |
|---|---|---|---|
| Critical radius constant $\gamma^d$ | $2^d(1+1/d) \cdot \mu(X_{\mathrm{free}})/\zeta_d$ | Same $/\varepsilon$ | $1/\varepsilon$ |
| Connection radius $r_n$ | $\gamma_* (\log n/n)^{1/d}$ | $\gamma_\varepsilon (\log n/n)^{1/d}$ | $\varepsilon^{-1/d}$ |
| Expected neighbours (sparse region) | $O(\log n)$ | $O(\log n)$ | $1$ |
| Expected neighbours (dense region) | $O(\log n)$ | $O(q_{\max}\mu \cdot \varepsilon^{-1} \log n)$ | Up to $q_{\max}\mu / \varepsilon$ |
| Effective uniform samples | $n$ | $\varepsilon n$ | $\varepsilon$ |
| Convergence rate to $c^*$ (worst case) | $\rho(n)$ | $\rho(\varepsilon n)$ | $\sim 1/\varepsilon$ slower |
| Probabilistic completeness | ✓ | ✓ | — |
| Asymptotic optimality | ✓ | ✓ | — |

---

## (e) Verdict

### Can S1-RRT\* be both theoretically consistent and meaningfully better than RRT\*?

**Yes.** The algorithm satisfies both requirements simultaneously:

1. **Theoretical consistency.** Probabilistic completeness and asymptotic optimality hold for *any* System 1 model $q$, including adversarial ones. The proofs require only $\varepsilon > 0$ (a fixed constant) and the adjusted connection radius $\gamma_\varepsilon = \gamma_* \cdot \varepsilon^{-1/d}$. No assumption on the quality, calibration, or coverage of $q$ is needed.

2. **Meaningful finite-budget improvement.** When $q$ concentrates mass near the optimal path, the algorithm finds good solutions in dramatically fewer iterations than uniform RRT\*, with an improvement ratio that grows as $\delta^{-d}$ in the corridor width. In moderate dimensions ($d \leq 6$) with $\varepsilon \approx 0.1$, the overhead of the guarantee (factor $\varepsilon^{-1/d} \approx 1.5$–$2.2$ in the connection radius; factor $1/\varepsilon = 10$ slowdown in the worst-case convergence rate) is modest compared to the potential orders-of-magnitude speedup from a good heuristic.

### Why the guarantee doesn't collapse

The structural reason is *separation of concerns*: the $\varepsilon$-uniform component is solely responsible for asymptotic coverage and optimality, while the $(1-\varepsilon) \cdot q$ component is solely responsible for finite-budget performance. These roles don't interfere because:
- Coverage depends only on a density *lower bound*, which the uniform component provides unconditionally.
- The covering-ball argument is *monotone*: adding more samples (from $q$) to a ball that already has a uniform sample doesn't hurt coverage; it only helps connectivity.
- S1's ordering heuristic is a permutation of an unchanged candidate set, so it cannot remove necessary connections.

### Honest caveats

1. **No free lunch.** If $q$ is bad, you pay a $1/\varepsilon$ convergence-rate penalty *and* the S1 inference cost per iteration. In the worst case, S1-RRT\* is strictly worse than RRT\* (by a constant factor).

2. **The mixture idea is not new.** The theoretical contribution is a clean derivation and the precise constant $\gamma_\varepsilon$, not a fundamentally new technique. The practical contribution is the integration framework for learned models.

3. **The k-nearest guarantee is conjectured.** While the radius-based proof is complete, the more practical k-nearest variant relies on a plausibility argument that we mark as a conjecture.

4. **Non-Euclidean / kinodynamic settings.** The proof assumes Euclidean cost and straight-line steering. Extension to kinodynamic systems (where the local connector is not simply Euclidean distance) requires additional work on the steer function and the cost monotonicity conditions, and is outside the scope of this analysis.

5. **Choice of $\varepsilon$.** There is a Pareto trade-off: small $\varepsilon$ maximises S1 utilisation but increases $\gamma_\varepsilon$ and worst-case slowdown; large $\varepsilon$ reduces S1's influence. For a given S1 model quality, the optimal $\varepsilon$ depends on the expected fraction of "useful" vs. "wasted" S1 samples, which is problem-dependent and generally unknown a priori. Adaptive selection of $\varepsilon$ (e.g. via bandit methods) is a natural extension but complicates the fixed-$\varepsilon$ theory.

### Final assessment

> [!TIP]
> S1-RRT\* is a clean, theoretically grounded way to inject learned priors into sampling-based planning. It is not a breakthrough — the core mixture-sampling idea is well-understood — but it is a *correct* and *practically useful* framework for deploying neural/VLM priors in RRT\* without sacrificing any formal guarantee. The price of robustness ($\varepsilon^{-1/d}$ radius inflation) is mild in low-to-moderate dimensions.

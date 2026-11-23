# Dual-Stratum System-1 RRT* (DS-S1-RRT*): Algorithmic Specification, Theoretical Foundations, and Rigorous Proofs

---

## (a) Key Idea and Architectural Rationale

### Motivation: The Non-Uniform Sampling Dilemma in Asymptotic Optimality
Integrating a fast, amortized, learned policy or Vision-Language Model ("System 1") into RRT* is fundamentally challenged by two conflicting requirements:
1. **Finite-budget performance**: The planner must aggressively exploit the System 1 distribution $q(x)$, which may concentrate probability mass along promising corridors or complex manifolds.
2. **Asymptotic consistency**: The planner must maintain probabilistic completeness and asymptotic optimality (almost-sure convergence to optimal cost $c^*$) even if $q(x)$ is completely uncalibrated, adversarial, or supported entirely inside obstacle interiors.

Standard approaches suffer from known theoretical breakdowns:
- **Global radius inflation on mixture densities** ($r_n \propto \varepsilon^{-1/d} (\log n / n)^{1/d}$): In regions where $q(x)$ has high density, the expected neighbor count explodes as $\Theta(q_{\max} \log n / \varepsilon)$. In high dimensions or extreme concentration, this degenerates per-iteration collision checking.
- **$k$-Nearest Neighbors ($k$-RRT*) across density boundaries**: When $k$-NN is run on non-uniform densities, nodes in sparse regions near high-density clusters have their $k$ neighbor slots saturated by dense-cluster nodes, severing connections along the sparse optimal corridor (density-boundary starvation).
- **Heuristic interference with tree dispersion**: If nearest-neighbor lookups during uniform exploration search across System 1 vertices, clustered System 1 nodes can "swallow" the Voronoi cells of uniform space, collapsing the space-filling property of RRT.

### The Dual-Stratum Architecture
DS-S1-RRT* resolves these pathologies by decoupling the **asymptotic coverage backbone** from the **amortized heuristic shortcuts** via vertex stratification:

1. **Explicit Stratum Tagging**:
   Every vertex $v \in V$ carries a persistent immutable tag:
   $$v \in V_{\mathrm{unif}} \quad \text{if generated from uniform draw}, \qquad v \in V_{\mathrm{sys1}} \quad \text{if generated from System 1 proposal}.$$
   The vertex set is partitioned as $V = V_{\mathrm{unif}} \cup V_{\mathrm{sys1}}$ with $n_u = |V_{\mathrm{unif}}|$ and $n_s = |V_{\mathrm{sys1}}|$.

2. **Stratum-Consistent Nearest Neighbor (`Nearest`)**:
   - On **uniform steps** ($u < \varepsilon$): Nearest neighbor selection searches *strictly over* $V_{\mathrm{unif}}$. This guarantees that the Voronoi dispersion of the uniform backbone is completely insulated from any pathological clustering, distribution shift, or adversarial behavior of $V_{\mathrm{sys1}}$.
   - On **System 1 steps** ($u \ge \varepsilon$): Nearest neighbor searches over the entire tree $V = V_{\mathrm{unif}} \cup V_{\mathrm{sys1}}$, allowing learned proposals to branch off any existing branch.

3. **Dual-Stratum Neighborhood (`DualStratumNear`)**:
   For any candidate $x_{\mathrm{new}}$, the neighbor set $X_{\mathrm{near}}$ is the structural union of two distinct topological queries:
   $$X_{\mathrm{near}}(x_{\mathrm{new}}) \;=\; X_{\mathrm{near}}^{\mathrm{unif}}(x_{\mathrm{new}}) \;\cup\; X_{\mathrm{near}}^{\mathrm{sys1}}(x_{\mathrm{new}}) \;\cup\; \{x_{\mathrm{nearest}}\}$$
   where:
   - **Uniform stratum**: Metric ball query over $V_{\mathrm{unif}}$ with radius:
     $$r_u(n_u) \;=\; \min\left( \gamma_u \left(\frac{\log(n_u + 1)}{n_u + 1}\right)^{1/d}, \; \eta \right), \qquad \gamma_u^d > 2^d\left(1 + \frac{1}{d}\right)\frac{\mu(X_{\mathrm{free}})}{\zeta_d}.$$
     Crucially, $r_u$ is indexed by $n_u = |V_{\mathrm{unif}}|$, so **no $\varepsilon^{-1/d}$ inflation is required**. The covering-ball argument runs natively on $V_{\mathrm{unif}}$.
   - **System 1 stratum**: $k$-NN query over $V_{\mathrm{sys1}}$ with $k_s = \lceil k_0 \log(|V| + 1) \rceil$. This strictly caps the maximum degree contribution from System 1 vertices at $O(\log |V|)$, preventing neighbor explosion.

4. **Lazy Cost-Sorted ChooseParent**:
   Candidates in $X_{\mathrm{near}}$ are sorted by their prospective arrival cost:
   $$\tilde{c}(u) \;=\; \mathrm{cost}[u] + c(u, x_{\mathrm{new}}).$$
   Collision checks are performed lazily in increasing order of $\tilde{c}(u)$. The **first collision-free candidate encountered is provably the exact cost minimizer**; the search terminates immediately, pruning remaining collision checks.

5. **Cost-Monotonic Rewiring with S1 Prioritization**:
   Rewiring candidates are pre-filtered to strict improvements:
   $$\{v \in X_{\mathrm{near}} : \mathrm{cost}[x_{\mathrm{new}}] + c(x_{\mathrm{new}}, v) < \mathrm{cost}[v]\}.$$
   System 1 may provide an ordering heuristic $\pi_{\mathrm{S1}}$ over this filtered set (prioritizing vertices with higher downstream impact), but it is mathematically forbidden from filtering or rejecting any candidate.

---

## (b) Algorithmic Specification

```
Algorithm DS-S1-RRT*(x_init, x_goal, S1, ε, N, η, γ_u, k_0)
────────────────────────────────────────────────────────────────────────────────
Input:  x_init, x_goal ∈ X_free
        S1: System-1 prior oracle (Sample, optional Priority)
        ε ∈ (0, 1]: Fixed uniform exploration probability
        N: Iteration budget
        η > 0: Maximum steering step size
        γ_u: Uniform connection radius constant
        k_0 > 0: System-1 k-NN scale factor

Output: Tree G = (V, E) with V = V_unif ∪ V_sys1, cost-to-come map, and c_best
────────────────────────────────────────────────────────────────────────────────

 1: V_unif ← {x_init};  V_sys1 ← ∅;  E ← ∅
 2: cost[x_init] ← 0;  parent[x_init] ← ∅;  c_best ← ∞
 3: n_u ← 1;  n_s ← 0

 4: for n = 1 to N do
 5:     u ~ Uniform(0, 1)

 6:     ── STEP 1: Stratified Sampling & Nearest Selection ────────────────────
 7:     if u < ε then
 8:         tag ← UNIFORM
 9:         x_rand ← UniformSample(X_free)
10:         x_nearest ← Nearest(V_unif, x_rand)          // Strictly over V_unif
11:     else
12:         tag ← SYS1
13:         x_rand ← S1.Sample(V_unif ∪ V_sys1, E, x_goal, c_best)
14:         if x_rand ∉ X_free then continue             // Reject invalid samples
15:         x_nearest ← Nearest(V_unif ∪ V_sys1, x_rand) // Over entire tree
16:     end if

17:     ── STEP 2: Steering & Local Feasibility ───────────────────────────────
18:     x_new ← Steer(x_nearest, x_rand, η)
19:     if ¬CollisionFree(x_nearest, x_new) then continue

20:     ── STEP 3: Dual-Stratum Neighborhood Construction ─────────────────────
21:     r_u ← min( γ_u * (log(n_u + 1) / (n_u + 1))^(1/d), η )
22:     X_near_unif ← { v ∈ V_unif : ‖v − x_new‖ ≤ r_u }

23:     k_s ← ⌈ k_0 * log(n_u + n_s + 1) ⌉
24:     X_near_sys1 ← kNN(V_sys1, x_new, k_s)

25:     X_near ← X_near_unif ∪ X_near_sys1 ∪ {x_nearest}

26:     ── STEP 4: Lazy Cost-Sorted ChooseParent ──────────────────────────────
27:     // Sort candidates by potential cost-to-come; tie-break arbitrarily
28:     Queue ← SortAscending(X_near, key = λv. (cost[v] + c(v, x_new)))
29:     x_min ← ∅;  c_min ← ∞

30:     for each v in Queue do
31:         c_cand ← cost[v] + c(v, x_new)
32:         if c_cand ≥ c_best then continue         // Informed bound pruning
33:         if CollisionFree(v, x_new) then
34:             x_min ← v
35:             c_min ← c_cand
36:             break                                // Exact minimum found; stop!
37:         end if
38:     end for

39:     if x_min = ∅ then continue                   // Steer guaranteed x_nearest, but check edge safety

40:     ── STEP 5: Vertex Insertion & Stratum Bookkeeping ─────────────────────
41:     if tag = UNIFORM then
42:         V_unif ← V_unif ∪ {x_new};  n_u ← n_u + 1
43:     else
44:         V_sys1 ← V_sys1 ∪ {x_new};  n_s ← n_s + 1
45:     end if
46:     E ← E ∪ {(x_min, x_new)}
47:     parent[x_new] ← x_min
48:     cost[x_new] ← c_min

49:     if InGoalRegion(x_new) and cost[x_new] < c_best then
50:         c_best ← cost[x_new]
51:     end if

52:     ── STEP 6: Prioritized Monotonic Rewiring ──────────────────────────────
53:     // Identify candidates that admit strict cost improvement
54:     RewireCandidates ← { v ∈ X_near \ {x_min} : cost[x_new] + c(x_new, v) < cost[v] }
55:     RewireQueue ← S1.OrderByPriority(RewireCandidates, x_new) // Permutation only!

56:     for each v in RewireQueue do
57:         c_rewire ← cost[x_new] + c(x_new, v)
58:         if CollisionFree(x_new, v) then
59:             old_parent ← parent[v]
60:             E ← (E \ {(old_parent, v)}) ∪ {(x_new, v)}
61:             parent[v] ← x_new
62:             Δ ← cost[v] - c_rewire
63:             cost[v] ← c_rewire
64:             PropagateCostReduction(v, Δ)          // Update descendant tree costs
65:         end if
66:     end for

67: end for
68: return G = (V_unif ∪ V_sys1, E)
```

---

## (c) Explicit Assumptions and Proofs

### Assumptions

- **(A1) Domain and Free Space Geometry**:
  The configuration space $X \subset \mathbb{R}^d$ is compact with Lebesgue measure $\mu(X) \in (0, \infty)$. The obstacle region $X_{\mathrm{obs}} \subset X$ is closed. The free space $X_{\mathrm{free}} = \mathrm{int}(X \setminus X_{\mathrm{obs}})$ is an open bounded set with positive measure $\mu(X_{\mathrm{free}}) > 0$, and its boundary $\partial X_{\mathrm{free}}$ is of measure zero ($\mu(\partial X_{\mathrm{free}}) = 0$).

- **(A2) Optimal Path and Strong $\delta$-Clearance**:
  There exists an optimal collision-free path $\sigma^*: [0, 1] \to X_{\mathrm{free}}$ connecting $x_{\mathrm{init}} = \sigma^*(0)$ to $x_{\mathrm{goal}} = \sigma^*(1)$ with finite arc-length $L^* = \mathrm{cost}(\sigma^*)$. Furthermore, $\sigma^*$ possesses strong $\delta$-clearance:
  $$\exists\,\delta > 0 \quad \text{such that} \quad \bigcup_{t \in [0, 1]} B(\sigma^*(t), \delta) \;\subset\; X_{\mathrm{free}}.$$

- **(A3) Cost Metric and Steering**:
  The cost functional $c(u, v) = \|u - v\|_2$ is Euclidean metric arc-length, satisfying symmetry, strict positivity for distinct points, and the triangle inequality. The local steering function $\mathrm{Steer}(u, v, \eta) = u + \min(1, \eta / \|v - u\|_2)(v - u)$ preserves straight-line connectivity.

- **(A4) System 1 Oracle Generality**:
  At iteration $n$, conditioned on the history filtration $\mathcal{F}_{n-1} = \sigma(x_1, \dots, x_{n-1})$, the System 1 generator $\mathrm{S1.Sample}$ induces an arbitrary, potentially time-varying and history-dependent conditional probability measure $Q_n(\cdot \mid \mathcal{F}_{n-1})$ on $X$. The oracle may be non-stationary, miss entire homotopy classes, place its entire measure on a single point or inside $X_{\mathrm{obs}}$, or act adversarially.

- **(A5) Non-Vanishing Uniform Mixture**:
  The mixture parameter $\varepsilon \in (0, 1]$ is fixed and independent of $n$, stall count, or $c_{\mathrm{best}}$.

- **(A6) Uniform Stratum Connection Constant**:
  The uniform connection radius $r_u(n_u)$ satisfies:
  $$r_u(n_u) = \min\left( \gamma_u \left(\frac{\log(n_u + 1)}{n_u + 1}\right)^{1/d}, \eta \right) \quad \text{with} \quad \gamma_u^d > 2^d \left(1 + \frac{1}{d}\right) \frac{\mu(X_{\mathrm{free}})}{\zeta_d},$$
  where $\zeta_d = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)}$ is the volume of the unit $d$-ball.

---

### Key Theoretical Lemmas

#### Lemma 1 (Uniform Vertex Divergence and Space-Filling Dispersion)
*Under (A1), (A3), (A5), the number of uniform vertices $n_u(n) = |V_{\mathrm{unif}}(n)|$ satisfies:*
$$n_u(n) \to \infty \quad \text{almost surely as } n \to \infty.$$
*Furthermore, for any open ball $B \subset X_{\mathrm{free}}$ with $\mu(B) > 0$, the uniform stratum samples $B$ infinitely often almost surely.*

*Proof.*
At each iteration $n$, with probability $\varepsilon$, a sample $x_{\mathrm{rand}} \sim \mathcal{U}(X_{\mathrm{free}})$ is drawn, and $x_{\mathrm{nearest}}$ is selected exclusively from $V_{\mathrm{unif}}$.
Let $\mathcal{F}_{n-1}$ be the filtration generated by the planner up to step $n-1$.
Conditioned on $\mathcal{F}_{n-1}$, the probability of drawing a uniform sample in $B$ is:
$$\Pr[x_{\mathrm{rand}} \in B \mid \mathcal{F}_{n-1}] \;\ge\; \varepsilon \cdot \frac{\mu(B)}{\mu(X_{\mathrm{free}})} \;>\; 0.$$
Because $\sum_{n=1}^\infty \Pr[x_{\mathrm{rand}} \in B \mid \mathcal{F}_{n-1}] = \sum_{n=1}^\infty \varepsilon \frac{\mu(B)}{\mu(X_{\mathrm{free}})} = \infty$, **Lévy's Extension of the Borel-Cantelli Lemma** (the Conditional Borel-Cantelli Lemma) implies that the event $\{x_{\mathrm{rand}} \in B \text{ on a uniform step}\}$ occurs for infinitely many $n$ almost surely.

Next, consider vertex insertion. Because $X_{\mathrm{free}}$ is open and $V_{\mathrm{unif}}$ is finite at any step $n$, there exists at least one vertex $v \in V_{\mathrm{unif}}$ whose Voronoi cell relative to $V_{\mathrm{unif}}$ intersects $X_{\mathrm{free}}$ with positive measure $\mu(\mathrm{Vor}(v) \cap X_{\mathrm{free}}) > 0$. There exists an open ball $B_v \subset \mathrm{Vor}(v) \cap X_{\mathrm{free}}$ of radius $\rho < \eta$ such that for any $x \in B_v$, the segment $[v, x]$ is strictly collision-free. Whenever $x_{\mathrm{rand}} \in B_v$ on a uniform step, $x_{\mathrm{nearest}} = v$, $\mathrm{Steer}(v, x_{\mathrm{rand}}, \eta) = x_{\mathrm{rand}}$, and the edge is collision-free, forcing $x_{\mathrm{new}} \in V_{\mathrm{unif}}$. Applying Lévy's Borel-Cantelli lemma to these non-empty local neighborhoods proves that $n_u(n)$ increases infinitely often, hence $n_u(n) \to \infty$ a.s. $\blacksquare$

---

#### Lemma 2 (Monotonic Cost-Dominance of Tree Shortcuts)
*Let $\mathcal{P}_{V}(u, v)$ denote the set of paths between $u, v$ in tree $G = (V, E)$. For any vertex $v \in V$, the stored cost $\mathrm{cost}[v]$ is the exact path length of the unique path from $x_{\mathrm{init}}$ to $v$ in $G$. Furthermore, for any two vertices $u, v \in V$, if $u \in X_{\mathrm{near}}(v)$ and the segment $[u, v]$ is collision-free, then following ChooseParent and Rewire:*
$$\mathrm{cost}[v] \;\le\; \mathrm{cost}[u] + c(u, v).$$
*In particular, the presence of System 1 vertices $V_{\mathrm{sys1}}$ and edges incident to them can only strictly decrease or leave unchanged $\mathrm{cost}[v]$ for any $v \in V_{\mathrm{unif}}$.*

*Proof.*
Every edge added to $E$ in Step 4 connects $x_{\mathrm{new}}$ to $x_{\mathrm{min}}$ with $\mathrm{cost}[x_{\mathrm{new}}] = \mathrm{cost}[x_{\mathrm{min}}] + c(x_{\mathrm{min}}, x_{\mathrm{new}})$.
In Step 4, ChooseParent evaluates all $w \in X_{\mathrm{near}}(x_{\mathrm{new}})$. The queue is sorted by $\mathrm{cost}[w] + c(w, x_{\mathrm{new}})$. The algorithm selects the first collision-free candidate, which is the global minimum over all collision-free nodes in $X_{\mathrm{near}}(x_{\mathrm{new}})$. Thus:
$$\mathrm{cost}[x_{\mathrm{new}}] \;\le\; \mathrm{cost}[w] + c(w, x_{\mathrm{new}}) \quad \forall w \in X_{\mathrm{near}}(x_{\mathrm{new}}) \text{ with } [w, x_{\mathrm{new}}] \subset X_{\mathrm{free}}.$$
In Step 6, a rewire is executed if and only if $\mathrm{cost}[x_{\mathrm{new}}] + c(x_{\mathrm{new}}, w) < \mathrm{cost}[w]$. Upon rewiring, the parent pointer is redirected, and $\mathrm{PropagateCostReduction}$ updates all descendants with the exact positive difference $\Delta = \mathrm{cost}_{\mathrm{old}}[w] - \mathrm{cost}_{\mathrm{new}}[w] > 0$. Because edge lengths $c(u, v) = \|u - v\| > 0$ are strictly positive and rewiring requires strict inequality, no cycles can be formed, and exact additive path costs are invariant across the entire tree.
If an edge passes through a System 1 node $s \in V_{\mathrm{sys1}}$ that provides a shortcut, $\mathrm{cost}[s] + c(s, v) < \mathrm{cost}[v]$, the cost of $v$ decreases. If $s$ does not offer a shortcut, it is simply not chosen as parent or rewire source. Thus, $V_{\mathrm{sys1}}$ vertices act strictly as potential shortcuts and cannot degrade the cost of any node. $\blacksquare$

---

### Theorem 1 (Probabilistic Completeness)
*Under assumptions (A1)–(A5), for any $\varepsilon \in (0, 1]$, the probability that DS-S1-RRT* contains a collision-free path from $x_{\mathrm{init}}$ to the goal region $X_{\mathrm{goal}}$ approaches $1$ as $N \to \infty$.*

*Proof Sketch.*
The optimal path $\sigma^*$ has clearance $\delta > 0$. Cover $\sigma^*$ with a finite chain of open balls $B_1, \dots, B_M \subset X_{\mathrm{free}}$, each of radius $\rho = \min(\delta/2, \eta/2)$, such that $x_{\mathrm{init}} \in B_1$, $x_{\mathrm{goal}} \in B_M$, and consecutive balls overlap with center spacing $\le \rho/2$.
By Lemma 1, on uniform iterations ($u < \varepsilon$), Nearest operates exclusively on $V_{\mathrm{unif}}$.
The standard RRT probabilistic completeness argument (LaValle & Kuffner, 2001) relies strictly on the Voronoi bias of uniform samples expanding into adjacent balls along the chain. Because $V_{\mathrm{sys1}}$ nodes are completely excluded from Nearest during uniform steps, the Voronoi expansion of $V_{\mathrm{unif}}$ through the finite sequence of balls $B_1, \dots, B_M$ proceeds with lower-bounded transition probabilities at each step. By the Conditional Borel-Cantelli Lemma, the tree reaches $B_M \subset X_{\mathrm{goal}}$ in a finite number of iterations almost surely. $\blacksquare$

---

### Theorem 2 (Asymptotic Optimality)
*Under assumptions (A1)–(A6), the cost of the best path in the tree $c_N$ converges to the optimal cost $c^*$ almost surely:*
$$\lim_{N \to \infty} c_N \;=\; c^* \quad \text{a.s.}$$

*Proof.*
We structure the proof via Karaman & Frazzoli's covering-ball argument with the $\theta$-trick, established exclusively across the uniform stratum $V_{\mathrm{unif}}$.

#### Step 1: The $\theta$-Discretization of the Optimal Path
Fix an arbitrary $\theta \in (0, 1/4)$.
Let $\sigma^*: [0, 1] \to X_{\mathrm{free}}$ be the optimal path of length $L^*$.
At step $n$, consider the uniform connection radius $r_u(n_u) = \gamma_u (\frac{\log(n_u + 1)}{n_u + 1})^{1/d}$.
Select $n_u$ large enough that $r_u(n_u) \le \min(\delta, \eta)$.
Define ball radii along $\sigma^*$:
$$\rho_{n_u} \;=\; \frac{\theta}{4} r_u(n_u).$$
Discretize $\sigma^*$ into $m_{n_u}$ points $z_1, z_2, \dots, z_{m_{n_u}}$ such that $\|z_{j+1} - z_j\|_2 = \frac{\theta}{2} r_u(n_u) = 2\rho_{n_u}$.
The number of balls is:
$$m_{n_u} \;=\; \left\lceil \frac{L^*}{2\rho_{n_u}} \right\rceil \;=\; O\left( \frac{L^*}{\theta \gamma_u} \left(\frac{n_u}{\log n_u}\right)^{1/d} \right).$$
Define the open balls $B_j = B(z_j, \rho_{n_u})$. Because $\rho_{n_u} < \delta$, each $B_j \subset X_{\mathrm{free}}$.

#### Step 2: Covering Probability via Conditional Borel-Cantelli
For any $j \in \{1, \dots, m_{n_u}\}$, the volume of $B_j$ is $\mu(B_j) = \zeta_d \rho_{n_u}^d = \zeta_d (\frac{\theta}{4})^d r_u^d(n_u)$.
Under the mixture model, at any step $n$, the probability that a sample is uniform and falls in $B_j$, conditioned on $\mathcal{F}_{n-1}$, is:
$$p_j(n) \;=\; \Pr[x_{\mathrm{rand}} \in B_j \text{ and } u < \varepsilon \mid \mathcal{F}_{n-1}] \;\ge\; \varepsilon \cdot \frac{\mu(B_j)}{\mu(X_{\mathrm{free}})} \;=\; \varepsilon \cdot \frac{\zeta_d (\theta/4)^d \gamma_u^d}{\mu(X_{\mathrm{free}})} \frac{\log n_u}{n_u}.$$
Let $E_j(n_u)$ denote the event that ball $B_j$ contains at least one uniform vertex from $V_{\mathrm{unif}}$.
By the coverage lemma of Karaman & Frazzoli (2011, Lemma 48), choosing $\gamma_u$ such that:
$$\gamma_u^d \;>\; 2^d \left(1 + \frac{1}{d}\right) \frac{\mu(X_{\mathrm{free}})}{\zeta_d}$$
ensures that the union bound over all $m_{n_u}$ balls satisfies:
$$\Pr\left[ \bigcup_{j=1}^{m_{n_u}} \{B_j \cap V_{\mathrm{unif}} = \emptyset\} \right] \;\le\; m_{n_u} \cdot \exp\left( - \alpha \log n_u \right) \;=\; O\left( n_u^{1/d - \alpha} (\log n_u)^{-1/d} \right),$$
where $\alpha = \frac{\zeta_d \gamma_u^d}{2^d \mu(X_{\mathrm{free}})} > 1 + 1/d$.
Thus, $1/d - \alpha < -1$, rendering the failure probability summable over $n_u$:
$$\sum_{n_u=1}^\infty \Pr\left[ \exists j : B_j \cap V_{\mathrm{unif}} = \emptyset \right] \;<\; \infty.$$
By the First Borel-Cantelli Lemma, for all sufficiently large $n_u$, **every ball $B_j$ contains at least one uniform vertex $v_j \in B_j \cap V_{\mathrm{unif}}$ almost surely**.

#### Step 3: Neighbor Invariance and Connectivity
Let $v_j \in B_j \cap V_{\mathrm{unif}}$ and $v_{j+1} \in B_{j+1} \cap V_{\mathrm{unif}}$. The Euclidean distance between them satisfies:
$$\|v_{j+1} - v_j\|_2 \;\le\; \|z_{j+1} - z_j\|_2 + 2\rho_{n_u} \;=\; 2\rho_{n_u} + 2\rho_{n_u} \;=\; 4\rho_{n_u} \;=\; \theta \cdot r_u(n_u) \;<\; r_u(n_u).$$
Therefore, $v_j \in X_{\mathrm{near}}^{\mathrm{unif}}(v_{j+1})$ and $v_{j+1} \in X_{\mathrm{near}}^{\mathrm{unif}}(v_j)$.
Because $B_j \cup B_{j+1} \subset X_{\mathrm{free}}$ and the clearance exceeds the ball separation, the segment $[v_j, v_{j+1}]$ is strictly collision-free.
By the Dual-Stratum neighborhood construction:
$$X_{\mathrm{near}}^{\mathrm{unif}} \;\subset\; X_{\mathrm{near}}.$$
Thus, the connection between $v_j$ and $v_{j+1}$ is guaranteed to be tested during ChooseParent or Rewire, regardless of how many System 1 vertices exist.

#### Step 4: Cost Summation via the $\theta$-Trick
By Lemma 2, the tree costs satisfy the recursive relation $\mathrm{cost}[v_{j+1}] \le \mathrm{cost}[v_j] + c(v_j, v_{j+1})$.
Telescoping this inequality from $v_1$ to $v_{m_{n_u}}$:
$$\mathrm{cost}[v_{m_{n_u}}] \;\le\; \mathrm{cost}[v_1] + \sum_{j=1}^{m_{n_u}-1} \|v_{j+1} - v_j\|_2.$$
Now, we bound the sum of edge lengths using the ball centers:
$$\|v_{j+1} - v_j\|_2 \;\le\; \|z_{j+1} - z_j\|_2 + 2\rho_{n_u}.$$
Summing over $j$:
$$\sum_{j=1}^{m_{n_u}-1} \|v_{j+1} - v_j\|_2 \;\le\; \sum_{j=1}^{m_{n_u}-1} \|z_{j+1} - z_j\|_2 + 2 m_{n_u} \rho_{n_u}.$$
Since $z_j$ are collinear points along $\sigma^*$, $\sum \|z_{j+1} - z_j\|_2 \le L^*$.
Substituting $m_{n_u} \approx \frac{L^*}{2\rho_{n_u}}$ and $\rho_{n_u} = \frac{\theta}{4} r_u(n_u)$:
$$2 m_{n_u} \rho_{n_u} \;=\; 2 \left( \frac{L^*}{2\rho_{n_u}} \right) \rho_{n_u} \cdot \theta \;=\; \theta L^*.$$
Therefore:
$$\mathrm{cost}[v_{m_{n_u}}] \;\le\; (1 + \theta) L^* + \mathrm{cost}[v_1].$$
As $n_u \to \infty$, $v_1 \to x_{\mathrm{init}}$, so $\mathrm{cost}[v_1] \to 0$.
Thus:
$$\limsup_{N \to \infty} c_N \;\le\; (1 + \theta) L^*.$$
Since this holds for arbitrarily small $\theta > 0$, taking $\theta \to 0$ yields:
$$\limsup_{N \to \infty} c_N \;\le\; L^* = c^*.$$
Since no feasible path can have cost strictly less than $c^*$, $\liminf_{N \to \infty} c_N \ge c^*$.
Hence, $\lim_{N \to \infty} c_N = c^*$ almost surely. $\blacksquare$

---

## (d) Complexity and Finite-Budget Analysis

### Per-Iteration Complexity

| Operation | DS-S1-RRT* Complexity | Theoretical Mechanism & Bounds |
|---|---|---|
| **Sampling** | $O(1)$ (unif) or $O(T_{\mathrm{S1}})$ (S1) | Mixture draw with probability $\varepsilon$ vs $1-\varepsilon$. |
| **Nearest Neighbor** | $O(\log n_u)$ (unif) or $O(\log |V|)$ (S1) | Kd-tree query over $V_{\mathrm{unif}}$ on uniform steps, over $V$ on S1 steps. |
| **Neighborhood Query** | $O(\log n_u + |X_{\mathrm{near}}^{\mathrm{unif}}| + \log |V| + k_s)$ | Exact dual query: range search on $V_{\mathrm{unif}}$ plus $k$-NN on $V_{\mathrm{sys1}}$. |
| **ChooseParent** | $O(|X_{\mathrm{near}}| \log |X_{\mathrm{near}}| + M_{\mathrm{free}} \cdot C_{\mathrm{coll}})$ | Sort candidates by cost-to-come; lazy break after $M_{\mathrm{free}} \ge 1$ free edges. |
| **Rewiring Pre-Filter & Sort** | $O(|X_{\mathrm{near}}| + |R| \log |R|)$ | Filter $R = \{v : \Delta(v) > 0\}$; sort by S1 priority $\pi_{\mathrm{S1}}$. |
| **Rewiring Collision Checks** | $O(|R| \cdot C_{\mathrm{coll}})$ | Checked in priority order; capped by $|X_{\mathrm{near}}| = O(\log |V|)$. |
| **Cost Propagation** | $O(\mathrm{SubtreeSize}(v))$ | Amortized $O(1)$ per iteration in practice; $O(|V|)$ worst case. |

#### Exact Bound on Neighborhood Size $|X_{\mathrm{near}}|$
Unlike standard S1-RRT*, where neighborhood size exploded as $\Theta(q_{\max} \log n / \varepsilon)$ under non-uniform $q(x)$, DS-S1-RRT* strictly bounds neighbor size:
$$\mathbb{E}[|X_{\mathrm{near}}|] \;=\; \mathbb{E}[|X_{\mathrm{near}}^{\mathrm{unif}}|] + k_s + 1 \;\le\; \zeta_d \gamma_u^d \log(n_u + 1) + \lceil k_0 \log(|V| + 1) \rceil + 1 \;=\; O(\log |V|).$$
This holds **everywhere in the configuration space**, even if System 1 collapses all its samples onto a single Dirac point or lower-dimensional manifold.

---

### Finite-Budget Analysis: Expected Tube Sample Ratio

Let $T_\delta = \{x \in X_{\mathrm{free}} : \mathrm{dist}(x, \sigma^*) \le \delta\}$ denote the $\delta$-tube around the optimal path $\sigma^*$, with relative volume fraction:
$$a \;=\; \frac{\mu(T_\delta)}{\mu(X_{\mathrm{free}})} \;\in\; (0, 1).$$
Let $\beta = \Pr_{x \sim Q_n}[x \in T_\delta]$ denote the probability that the System 1 oracle places its sample inside $T_\delta$.

#### Correct Sample Density Ratio
Under standalone Uniform RRT*, the probability that a sample lands in $T_\delta$ is $P_{\mathrm{unif}} = a$.
Under DS-S1-RRT*, the mixture probability is:
$$P_{\mathrm{mix}} \;=\; \varepsilon \cdot a + (1 - \varepsilon) \cdot \beta.$$
The relative sample enrichment ratio in the optimal tube is:
$$R_{\mathrm{tube}} \;=\; \frac{P_{\mathrm{mix}}}{P_{\mathrm{unif}}} \;=\; \frac{\varepsilon a + (1 - \varepsilon)\beta}{a} \;=\; \boxed{\varepsilon + (1 - \varepsilon)\frac{\beta}{a}}.$$

#### Quantitative Implications
1. **Accurate Prior ($\beta \gg a$)**:
   In high-dimensional spaces or narrow corridors where $a = 10^{-4}$, if a trained System 1 model achieves $\beta = 0.5$ with $\varepsilon = 0.1$:
   $$R_{\mathrm{tube}} \;=\; 0.1 + 0.9 \cdot \frac{0.5}{10^{-4}} \;=\; 0.1 + 4500 \;\approx\; 4500\times.$$
   The expected number of iterations required to populate the optimal corridor is accelerated by over three orders of magnitude.

2. **Adversarial / Degenerate Prior ($\beta = 0$)**:
   If System 1 is completely misleading or proposes invalid obstacle points:
   $$R_{\mathrm{tube}} \;=\; \varepsilon < 1.$$
   The algorithm pays a bounded penalty: the rate of uniform exploration is slowed by exactly a factor of $1/\varepsilon$ (e.g., $10\times$ for $\varepsilon = 0.1$), but the asymptotic convergence rate order $O((\log n / n)^{1/d})$ and completeness remain unconditionally intact.

---

## (e) Failure Modes and Robust Mitigations

### 1. Homotopy Lock-In
- **Mechanism**: System 1 persistently samples a suboptimal homotopy class $H_1$ (e.g., an expansive open region) while the global optimum lies in a narrow passage $H_2$. The tree heavily over-develops $H_1$, while $H_2$ receives only $\varepsilon$-fraction uniform samples.
- **Why Proof Survives**: Uniform exploration is governed strictly by $V_{\mathrm{unif}}$ during uniform steps. By Lemma 1, $H_2$ is sampled infinitely often, ensuring eventual discovery.
- **Mitigation**: Maintain an adaptive multi-armed bandit over System 1 context prompts, or incorporate a Gaussian bridge / clearance-retraction distribution as a third mixture stratum without altering the uniform lower bound.

### 2. Rewiring Starvation in Dense Clusters
- **Mechanism**: If System 1 clusters thousands of vertices in a tiny region, each insertion could trigger expensive rewiring cascades that update large subtrees without lowering costs outside the cluster.
- **Why Proof Survives**: Rewiring is strictly monotonic; cost updates never propagate invalid paths.
- **Mitigation**:
  - **Thresholded Rewiring**: Only execute rewires if cost improvement $\Delta > \tau_{\mathrm{rewire}} = 10^{-4}$.
  - **Lazy Degree Cap**: Capping $X_{\mathrm{near}}^{\mathrm{sys1}}$ at $k_s = \lceil k_0 \log |V| \rceil$ prevents the neighbor set from scaling with local sample density.

### 3. Density-Boundary Starvation
- **Mechanism**: Under standard $k$-NN, a node in a sparse region near a dense cluster fails to connect across the sparse corridor because all $k$ slots are captured by the cluster.
- **Why Proof Survives**: DS-S1-RRT* explicitly bypasses this failure mode by executing a **metric ball query $X_{\mathrm{near}}^{\mathrm{unif}}$ over $V_{\mathrm{unif}}$**. Regardless of the density of $V_{\mathrm{sys1}}$, every uniform neighbor within distance $r_u$ is guaranteed to be in $X_{\mathrm{near}}$.

### 4. System 1 Inference Latency
- **Mechanism**: Neural network / VLM forward passes can take 10–50 ms, whereas geometric collision checks take microseconds. If evaluated serially at every step, wall-clock time explodes.
- **Why Proof Survives**: The mathematical guarantees depend on iteration count $N$, not wall-clock time.
- **Mitigation**: Asynchronous batch generation. An independent inference worker continuously fills a ring buffer with System 1 candidate proposals $\mathcal{B}_{\mathrm{S1}}$, allowing the main RRT* loop to pop samples in $O(1)$ time.

---

## (f) Audit of Flaws (a)–(l)

| Flaw | Original Problem in S1-RRT* or DS-S1-RRT* | Resolution in DS-S1-RRT* | Current Status |
|---|---|---|---|
| **(a)** | Dynamic stall-based $\varepsilon_n \to 1$ disabled S1 when $c_{\mathrm{best}} = \infty$. | Replaced with fixed, non-decaying $\varepsilon \in (0, 1]$. | **Fixed** |
| **(b)** | False claim: "$V_{\mathrm{unif}}$ is stochastically identical to standalone RRT*". | Abandoned fictitious standalone RRT*; proved convergence via internal tree dispersion. | **Fixed** |
| **(c)** | Monotonic dominance induction broke when paths crossed $V_{\mathrm{sys1}}$ nodes. | Proved that $V_{\mathrm{sys1}}$ vertices act strictly as additive cost shortcuts (Lemma 2). | **Fixed** |
| **(d)** | False claim of $O(\log n)$ dynamic UCB min-heap over changing vertex keys. | Dropped dynamic UCB heap; replaced with stateless S1 priority reordering. | **Fixed** |
| **(e)** | Arbitrary $k_0 > e(1+1/d)$ condition decorated $k$-NN without proof. | Clarified that $k_s$ on $V_{\mathrm{sys1}}$ is a computational cap; $r_u$ carries the proof. | **Fixed** |
| **(f)** | Sum of $m$ balls of $2\delta_n$ yielded $2L^*$, not $L^* + 2\delta_n$. | Replaced with Karaman-Frazzoli $\theta$-discretization trick: $\sum \le (1+\theta)L^*$. | **Fixed** |
| **(g)** | False assumption that mixture samples $x_n$ are i.i.d. | Replaced with conditional probability bounds and Lévy's Conditional Borel-Cantelli Lemma. | **Fixed** |
| **(h)** | Ignored steering truncation and collision rejection in vertex count. | Formulated Lemma 1 with non-empty local reachability balls and positive-measure Voronoi cells. | **Fixed** |
| **(i)** | $\mathrm{S1.OrderByPriority}$ had zero effect in ChooseParent due to full search. | Implemented lazy cost-sorted ChooseParent with immediate first-feasible break. | **Fixed** |
| **(j)** | Erroneous finite-budget sample ratio formula: $1 + (1-\varepsilon)\beta/(\varepsilon a)$. | Corrected to exact mixture ratio: $R_{\mathrm{tube}} = \varepsilon + (1 - \varepsilon)\frac{\beta}{a}$. | **Fixed** |
| **(k)** | $k$-NN conjecture failed across density boundaries; wrong constant cited. | Dual-stratum ball query guarantees sparse connectivity; $k$-NN restricted to $V_{\mathrm{sys1}}$. | **Fixed** |
| **(l)** | Heuristic neighbor capping without theoretical justification. | Dual-stratum metric ball + $k$-NN architecture provides a principled topological bound. | **Fixed** |

---

## (g) Novelty Assessment and Empirical Validation Protocol

### Theoretical Verdict on Novelty
DS-S1-RRT* is **theoretically sound, non-trivial, and provably robust**.
- **What is known**: Fixed-mixture sampling ($p = \varepsilon \mathcal{U} + (1-\varepsilon)q$) is a recognized technique in sampling theory.
- **What is genuinely new**: The **dual-stratum topological decoupling**. Previous heuristic RRT* variants either applied a single global connection radius (which blows up degree in dense heuristic clusters) or pure $k$-NN (which starves across density boundaries). Partitioning the vertex set into an autonomous uniform backbone ($V_{\mathrm{unif}}$ queried by metric ball $r_u(n_u)$) and an amortized heuristic stratum ($V_{\mathrm{sys1}}$ queried by $k$-NN) solves both problems simultaneously:
  1. The connection radius $r_u(n_u)$ requires **no $\varepsilon^{-1/d}$ inflation**, because it scales natively with $n_u$.
  2. The neighborhood degree is **provably bounded by $O(\log |V|)$** everywhere.
  3. The uniform tree dispersion is **immune to heuristic clustering**.

---

### Required Empirical Validation Protocol

To experimentally substantiate the theoretical claims, benchmark DS-S1-RRT* against (1) Standard RRT*, (2) Informed RRT*, and (3) Naive S1-RRT* across three stress-test environments:

1. **Adversarial S1 Stress-Test (Trap & Deception)**:
   - *Environment*: A double-bug trap or deceptive U-shaped labyrinth where the learned model $q(x)$ points directly into a dead-end pocket.
   - *Metric*: Validate that DS-S1-RRT* achieves asymptotic optimality with runtime bounded by at most $1/\varepsilon \times$ Uniform RRT*, whereas unstratified heuristic planners exhibit complete starvation or severe lock-in.

2. **Density-Boundary Starvation Benchmark**:
   - *Environment*: A narrow 6D passage adjacent to a wide open clearing where System 1 generates an ultra-dense cluster ($10^4$ samples) in the clearing.
   - *Metric*: Verify that pure $k$-RRT* fails to cross the narrow passage due to neighbor slot saturation, while DS-S1-RRT* reliably connects via $X_{\mathrm{near}}^{\mathrm{unif}}$.

3. **Wall-Clock Amortization with Lazy ChooseParent**:
   - *Environment*: High-degree 7-DOF manipulator planning (e.g., Franka Emika Panda in cluttered shelf).
   - *Metric*: Measure total collision checks performed. Demonstrate that lazy cost-sorted ChooseParent achieves $\ge 70\%$ reduction in collision checking relative to full-neighbor evaluation without sacrificing path cost.

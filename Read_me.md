# Part 2 — Factory & Belts

This repository contains two small command-line tools required for the ERP.AI engineering assessment: a factory steady-state solver and a bounded-belts (flow) solver. Both tools strictly accept JSON on stdin and produce JSON on stdout with no extra prints.

## Design notes (high level)

### Factory (factory/main.py)
- Modeling: We model each recipe's crafts/min as a decision variable `x_r >= 0`.
- Conservation: For each item `i` we enforce

  `sum_r out_r[i] * (1+prod_m) * x_r  - sum_r in_r[i] * x_r = b[i]`

  where `b[target]` equals the requested rate, intermediates are balanced (`0`), and raw items are constrained to be net-consumed and within caps.
- Machine caps: Each machine type `m` limits `sum_{r on m} x_r / eff_crafts_per_min(r) <= max_machines[m]`.
-  Modules: `speed` modifies effective crafts/min per recipe; `prod` multiplies outputs only. Module settings are read per machine type and applied uniformly to all recipes that run on that machine.
- Objective/Tie-break: The LP minimizes total machines (a linear objective). To keep outputs deterministic, a tiny lexicographic tie-break (based on recipe ordering) is added.
- Infeasibility handling: If the requested target is infeasible, the solver solves a second LP to maximize feasible target `T`. The solver returns `max_feasible_target_per_min` and simple bottleneck hints (tight machine caps and raw supplies).
- Solver: `pulp` (CBC backend) — chosen for reliability and reproducibility. See `RUN.md` for install instructions.

### Belts (belts/main.py)
- Lower bounds: We implement the standard lower-bound to maxflow transformation by reducing capacities to `hi-lo` and accumulating node imbalances.
- Node caps: Nodes with capacity constraints are handled with node-splitting (`v_in -> v_out` with capacity cap).
- Feasibility strategy: Create a super-source/sink to satisfy node imbalances and run a maxflow. If the lower-bound check passes, run a second maxflow to route actual supplies to sink.
- Infeasibility certificate: When infeasible, the residual reachable set and tight edges/nodes crossing the cut are reported per the assignment schema.
- Determinism: The implementation uses a deterministic Dinic implementation and consistent iteration orders.

## Numeric tolerances
- Absolute tolerance for balances and caps: `1e-9`.
- Small tie-break epsilon in factory LP: `1e-7`.

## Running locally
- Install dependencies:

```bash
pip install pulp


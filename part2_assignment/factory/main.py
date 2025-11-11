#!/usr/bin/env python3
"""
Factory steady-state solver (CLI)
Reads JSON from stdin and writes JSON to stdout. Uses pulp for LP solving.
Usage: python factory/main.py < input.json > output.json
"""
import sys
import json
from math import isclose

# pulp is required
try:
    import pulp
except Exception as e:
    sys.stderr.write("Missing dependency: pulp. Install with: pip install pulp\n")
    raise

TOL = 1e-9


def main():
    data = json.load(sys.stdin)

    machines = data.get("machines", {})
    recipes = data.get("recipes", {})
    modules = data.get("modules", {})
    limits = data.get("limits", {})
    target = data.get("target", {})

    raw_supply = limits.get("raw_supply_per_min", {})
    max_machines = limits.get("max_machines", {})

    target_item = target.get("item")
    requested_rate = float(target.get("rate_per_min", 0.0))

    # normalize structures
    # machine info: crafts_per_min
    for m, info in machines.items():
        if "crafts_per_min" not in info:
            raise ValueError(f"machine {m} missing crafts_per_min")

    # compute effective per-recipe numbers
    recipe_names = sorted(recipes.keys())

    # map recipe -> machine, time_s, ins, outs
    recipe_info = {}
    items = set()
    for r in recipe_names:
        info = recipes[r]
        m = info["machine"]
        time_s = float(info["time_s"])
        ins = {k: float(v) for k, v in info.get("in", {}).items()}
        outs = {k: float(v) for k, v in info.get("out", {}).items()}
        recipe_info[r] = {
            "machine": m,
            "time_s": time_s,
            "in": ins,
            "out": outs,
        }
        items.update(ins.keys())
        items.update(outs.keys())

    items = sorted(items)

    # module values per machine
    module_speed = {m: float(modules.get(m, {}).get("speed", 0.0)) for m in machines}
    module_prod = {m: float(modules.get(m, {}).get("prod", 0.0)) for m in machines}

    # effective crafts/min per recipe
    eff_crafts = {}
    for r in recipe_names:
        m = recipe_info[r]["machine"]
        base = float(machines[m]["crafts_per_min"])
        speed = module_speed.get(m, 0.0)
        time_s = recipe_info[r]["time_s"]
        eff = base * (1.0 + speed) * 60.0 / time_s
        eff_crafts[r] = eff

    # Prepare LP
    # Variables: x_r >= 0 for each recipe
    # Objective: minimize total_machines = sum_r x_r / eff_crafts[r]
    # Conservation constraints per item

    prob = pulp.LpProblem("factory_min_machines", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", recipe_names, lowBound=0, cat="Continuous")

    # build conservation expressions
    # For each item i: expr_i = sum_r out_i*(1+prod_m_r)*x_r - sum_r in_i * x_r
    item_expr = {}
    for i in items:
        expr = None
        expr = pulp.LpAffineExpression()
        for r in recipe_names:
            outs = recipe_info[r]["out"]
            ins = recipe_info[r]["in"]
            m = recipe_info[r]["machine"]
            prod = module_prod.get(m, 0.0)
            out_q = outs.get(i, 0.0) * (1.0 + prod)
            in_q = ins.get(i, 0.0)
            if out_q:
                expr += out_q * x[r]
            if in_q:
                expr += -in_q * x[r]
        item_expr[i] = expr

    # Apply constraints
    # target equality
    if target_item is None:
        out = {"status": "infeasible", "max_feasible_target_per_min": 0.0, "bottleneck_hint": ["no target specified"]}
        json.dump(out, sys.stdout)
        return

    if target_item not in item_expr:
        # target not producible by recipes -> infeasible immediately
        out = {"status": "infeasible", "max_feasible_target_per_min": 0.0, "bottleneck_hint": ["target not producible by any recipe"]}
        json.dump(out, sys.stdout)
        return

    prob += 0  # placeholder objective; we will set after building machine expression

    # Raw item constraints: for raw items present in raw_supply
    raw_items = set(raw_supply.keys())

    # Add item constraints for intermediates: equality zero
    for i in items:
        expr = item_expr[i]
        if i == target_item:
            # equality to requested rate for feasibility/minimize machines phase
            prob += (expr == requested_rate)
        elif i in raw_items:
            # raw items must be net-consumed only and within cap: expr <= 0 and expr >= -cap
            cap = float(raw_supply[i])
            prob += (expr <= 0.0)
            # expr >= -cap  -> -expr <= cap
            prob += (-expr <= cap + TOL)
        else:
            # intermediate: balanced
            prob += (expr == 0.0)

    # Machine caps
    for m, info in machines.items():
        cap = float(max_machines.get(m, float('inf')))
        # sum_{r uses m} x_r / eff_crafts[r] <= cap
        expr = pulp.LpAffineExpression()
        for r in recipe_names:
            if recipe_info[r]["machine"] == m:
                eff = eff_crafts[r]
                expr += (1.0 / eff) * x[r]
        prob += (expr <= cap + TOL)

    # Objective: minimize total machines; add tiny lexicographic tie-break stable by recipe name
    total_machines_expr = pulp.LpAffineExpression()
    for idx, r in enumerate(recipe_names):
        total_machines_expr += (1.0 / eff_crafts[r]) * x[r]
    # tiny lexicographic tiebreak
    eps = 1e-7
    tie_expr = pulp.LpAffineExpression()
    for idx, r in enumerate(recipe_names):
        tie_expr += (idx + 1) * x[r]
    prob.sense = pulp.LpMinimize
    prob.setObjective(total_machines_expr + eps * tie_expr)

    # Solve for feasibility + minimize machines
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    pulp_status = pulp.LpStatus.get(status, pulp.LpStatus[prob.status])

    if pulp_status == 'Optimal' or pulp_status == 'Feasible':
        # extract solution
        per_recipe = {r: float(pulp.value(x[r])) for r in recipe_names}
        per_machine = {}
        for m in machines:
            used = 0.0
            for r in recipe_names:
                if recipe_info[r]["machine"] == m:
                    eff = eff_crafts[r]
                    used += per_recipe[r] / eff
            per_machine[m] = used
        # compute raw consumption per minute
        raw_consumption = {}
        for raw in raw_items:
            val = 0.0
            for r in recipe_names:
                ins = recipe_info[r]["in"]
                outs = recipe_info[r]["out"]
                m = recipe_info[r]["machine"]
                prod = module_prod.get(m, 0.0)
                out_q = outs.get(raw, 0.0) * (1.0 + prod)
                in_q = ins.get(raw, 0.0)
                val += out_q * per_recipe[r] - in_q * per_recipe[r]
            # raw items are net-consumed only, take abs of negative
            raw_consumption[raw] = abs(min(0.0, val))

        out = {
            "status": "ok",
            "per_recipe_crafts_per_min": per_recipe,
            "per_machine_counts": per_machine,
            "raw_consumption_per_min": raw_consumption,
        }
        json.dump(out, sys.stdout)
        return

    # If we reach here: infeasible for requested_rate. Find max feasible target by maximizing T
    # Build a new LP: variables x_r >=0 and T >=0
    prob2 = pulp.LpProblem("factory_max_target", pulp.LpMaximize)
    x2 = pulp.LpVariable.dicts("x", recipe_names, lowBound=0, cat="Continuous")
    Tvar = pulp.LpVariable("T", lowBound=0, cat="Continuous")

    # Conservation constraints
    for i in items:
        expr = pulp.LpAffineExpression()
        for r in recipe_names:
            outs = recipe_info[r]["out"]
            ins = recipe_info[r]["in"]
            m = recipe_info[r]["machine"]
            prod = module_prod.get(m, 0.0)
            out_q = outs.get(i, 0.0) * (1.0 + prod)
            in_q = ins.get(i, 0.0)
            if out_q:
                expr += out_q * x2[r]
            if in_q:
                expr += -in_q * x2[r]
        if i == target_item:
            prob2 += (expr == Tvar)
        elif i in raw_items:
            cap = float(raw_supply[i])
            prob2 += (expr <= 0.0)
            prob2 += (-expr <= cap + TOL)
        else:
            prob2 += (expr == 0.0)

    # machine caps
    for m, info in machines.items():
        cap = float(max_machines.get(m, float('inf')))
        expr = pulp.LpAffineExpression()
        for r in recipe_names:
            if recipe_info[r]["machine"] == m:
                eff = eff_crafts[r]
                expr += (1.0 / eff) * x2[r]
        prob2 += (expr <= cap + TOL)

    prob2 += Tvar
    status2 = prob2.solve(solver)
    pulp_status2 = pulp.LpStatus.get(status2, pulp.LpStatus[prob2.status])

    if pulp_status2 not in ('Optimal', 'Feasible'):
        # completely infeasible system
        out = {"status": "infeasible", "max_feasible_target_per_min": 0.0, "bottleneck_hint": ["no feasible production"]}
        json.dump(out, sys.stdout)
        return

    T_max = float(pulp.value(Tvar))

    # Now, to provide bottleneck hints: inspect machine caps and raw caps that are tight at this solution
    per_recipe_sol = {r: float(pulp.value(x2[r])) for r in recipe_names}

    tight = []
    # machine tightness
    for m in machines:
        cap = float(max_machines.get(m, float('inf')))
        used = 0.0
        for r in recipe_names:
            if recipe_info[r]["machine"] == m:
                used += per_recipe_sol[r] / eff_crafts[r]
        if used >= cap - 1e-6:
            tight.append(m + " cap")

    # raw tightness
    for raw in raw_items:
        val = 0.0
        for r in recipe_names:
            ins = recipe_info[r]["in"]
            outs = recipe_info[r]["out"]
            m = recipe_info[r]["machine"]
            prod = module_prod.get(m, 0.0)
            out_q = outs.get(raw, 0.0) * (1.0 + prod)
            in_q = ins.get(raw, 0.0)
            val += out_q * per_recipe_sol[r] - in_q * per_recipe_sol[r]
        consumption = abs(min(0.0, val))
        cap = float(raw_supply.get(raw, 0.0))
        if consumption >= cap - 1e-6:
            tight.append(raw + " supply")

    out = {
        "status": "infeasible",
        "max_feasible_target_per_min": T_max,
        "bottleneck_hint": tight,
    }
    json.dump(out, sys.stdout)


if __name__ == '__main__':
    main()

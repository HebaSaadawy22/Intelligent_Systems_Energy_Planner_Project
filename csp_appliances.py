"""Appliance scheduling CSP: Backtracking, Forward Checking, AC-3 (switchable).

Variable  : appliance start hour
Domain    : start hours inside the appliance window where forecast + power <= limit
Constraints (binary, used by AC-3 and FC):
    - order      : dryer starts after washer has finished
    - circuit    : appliances on the same circuit must not overlap
    - pair load  : forecast + both appliances <= limit in overlapping hours
Global: total load (forecast + all appliances) <= limit in every hour.
Forbidden combinations (feedback from battery search) are rejected.
Counters: assignments, backtracks, ac3_revisions.
"""
from __future__ import annotations

from collections import deque

EPS = 1e-9


class ApplianceCSP:
    def __init__(self, forecast, appliances, limit, circuits=None,
                 order_constraints=None, forbidden=None):
        self.forecast = list(forecast)
        self.H = len(self.forecast)
        self.app = appliances
        self.limit = limit
        self.names = list(appliances)
        self.order = list(order_constraints or [])
        self.forbidden = [dict(f) for f in (forbidden or [])]
        self.circuit_of = {}
        for cname, members in (circuits or {}).items():
            for m in members:
                self.circuit_of.setdefault(m, set()).add(cname)
        self.reset_counters()

    # ---------- counters ----------
    def reset_counters(self):
        self.assignments = 0
        self.backtracks = 0
        self.ac3_revisions = 0

    # ---------- helpers ----------
    def _span(self, name, start):
        return range(start, start + self.app[name]["duration"])

    def initial_domain(self, name):
        a = self.app[name]
        dom = []
        for s in range(a["earliest"], a["latest_end"] - a["duration"] + 1):
            if all(self.forecast[h] + a["power"] <= self.limit + EPS
                   for h in self._span(name, s)):
                dom.append(s)
        return dom

    def unfit_appliances(self):
        """Appliances whose window cannot fit even alone."""
        return [n for n in self.names if not self.initial_domain(n)]

    def _binary_ok(self, a, va, b, vb):
        da, db = self.app[a]["duration"], self.app[b]["duration"]
        for before, after in self.order:
            if a == before and b == after and vb < va + da:
                return False
            if a == after and b == before and va < vb + db:
                return False
        if self.circuit_of.get(a, set()) & self.circuit_of.get(b, set()):
            if not (va + da <= vb or vb + db <= va):
                return False
        pa, pb = self.app[a]["power"], self.app[b]["power"]
        for h in set(self._span(a, va)) & set(self._span(b, vb)):
            if self.forecast[h] + pa + pb > self.limit + EPS:
                return False
        return True

    def _load_ok(self, assign, name, val):
        load = list(self.forecast)
        for n, s in assign.items():
            for h in self._span(n, s):
                load[h] += self.app[n]["power"]
        for h in self._span(name, val):
            load[h] += self.app[name]["power"]
        return all(x <= self.limit + EPS for x in load)

    def _forbidden_hit(self, assign):
        return any(all(assign.get(k) == v for k, v in f.items())
                   for f in self.forbidden)

    def _consistent(self, assign, name, val):
        for other, ov in assign.items():
            if not self._binary_ok(name, val, other, ov):
                return False
        if not self._load_ok(assign, name, val):
            return False
        return not self._forbidden_hit({**assign, name: val})

    # ---------- AC-3 ----------
    def ac3(self, domains):
        queue = deque((a, b) for a in self.names for b in self.names if a != b)
        while queue:
            a, b = queue.popleft()
            self.ac3_revisions += 1
            removed = False
            for va in list(domains[a]):
                if not any(self._binary_ok(a, va, b, vb) for vb in domains[b]):
                    domains[a].remove(va)
                    removed = True
            if removed:
                if not domains[a]:
                    return False
                for c in self.names:
                    if c not in (a, b):
                        queue.append((c, a))
        return True

    # ---------- search ----------
    def solve(self, use_fc=False, use_ac3=False):
        self.reset_counters()
        domains = {n: self.initial_domain(n) for n in self.names}
        if any(not d for d in domains.values()):
            return None
        if use_ac3 and not self.ac3(domains):
            return None
        return self._bt({}, domains, use_fc)

    def _bt(self, assign, domains, use_fc):
        if len(assign) == len(self.names):
            return dict(assign)
        var = next(n for n in self.names if n not in assign)
        for val in domains[var]:
            self.assignments += 1
            if not self._consistent(assign, var, val):
                continue
            assign[var] = val
            new_domains = domains
            ok = True
            if use_fc:
                new_domains = {n: list(d) for n, d in domains.items()}
                new_domains[var] = [val]
                for other in self.names:
                    if other in assign:
                        continue
                    new_domains[other] = [v for v in domains[other]
                                          if self._consistent(assign, other, v)]
                    if not new_domains[other]:
                        ok = False
                        break
            if ok:
                res = self._bt(assign, new_domains, use_fc)
                if res is not None:
                    return res
            del assign[var]
            self.backtracks += 1
        return None

    def load_profile(self, schedule):
        load = list(self.forecast)
        for n, s in schedule.items():
            for h in self._span(n, s):
                load[h] += self.app[n]["power"]
        return load


def plan_appliances(forecast, appliances, limit, circuits=None,
                    order_constraints=None, forbidden=None,
                    use_fc=True, use_ac3=True, relax_steps=5, relax_step=0.10):
    """Solve; if impossible, relax the limit in 10% steps and report the violation."""
    unfit = None
    for k in range(relax_steps + 1):
        lim = limit * (1 + relax_step * k)
        csp = ApplianceCSP(forecast, appliances, lim, circuits,
                           order_constraints, forbidden)
        sol = csp.solve(use_fc=use_fc, use_ac3=use_ac3)
        if k == 0:
            unfit = csp.unfit_appliances()
        if sol is not None:
            return {
                "status": "ok" if k == 0 else "relaxed",
                "schedule": sol,
                "limit_used": lim,
                "violation_kw": max(0.0, lim - limit),
                "relax_percent": round(100 * relax_step * k),
                "counters": {"assignments": csp.assignments,
                             "backtracks": csp.backtracks,
                             "ac3_revisions": csp.ac3_revisions},
            }
    return {
        "status": "no_solution",
        "message": (f"NO SOLUTION: appliances cannot be scheduled under {limit} kW "
                    f"even after relaxing the limit up to "
                    f"{round(100 * relax_step * relax_steps)}%."),
        "windows_that_cannot_fit": unfit,
    }

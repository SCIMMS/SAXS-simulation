"""Bounded supplementary penalty calibration; all prior evidence is read-only.

Extends the exact old calibration draw, retains zero, and compares the explicit
zero-extra-mode limit. Test data and known shape truth never select a penalty.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import revision_differentiable_synthetic as syn
from revision_common import ROOT, dump, save_csv, sha, np
from scipy.optimize import least_squares, nnls

OUT = ROOT / "output/differentiable_si_validation_2026-10-12/penalty_selection"
PRIOR = ROOT / "output/differentiable_synthetic_2026-10-11"
EXTRA_LAMBDAS = [1e4, 1e5, 1e6]


def protected():
    paths = [ROOT / "manuscript_rewrite_txt/M Kim et al - JSR v01-R9.docx",
             ROOT / "uploaded_SAXS_simulator/saxs_module.py",
             ROOT / "validation/revision_differentiable_synthetic.py"]
    paths += [p for p in PRIOR.rglob("*") if p.is_file()]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def bulk_limit(forward, observed, mask, starts):
    """Fit the same six bulk/nuisance variables with all extra modes fixed at 0."""
    lower = np.r_[np.log([17., 17., 1.7]), -.5, np.full(len(syn.MODES), -.10), -35., np.log(.01)]
    upper = np.r_[np.log([34., 34., 16.]), .65, np.full(len(syn.MODES), .10), 5., np.log(5.)]
    idx = np.r_[np.arange(4), len(syn.NAMES)-2, len(syn.NAMES)-1]
    records = []
    for k, start in enumerate(starts):
        start = np.asarray(start, float).copy()
        start[4:-2] = 0
        raw = np.abs(forward.amplitude(start, derivatives=False)[0])**2
        norm = float(raw[mask].max())
        nuisance, _ = nnls(np.c_[raw[mask]/norm, np.ones(mask.sum())], observed[mask])
        start[-2] = np.log(max(nuisance[0]/norm, 1e-15))
        start[-1] = np.log(np.clip(nuisance[1], .011, 4.9))
        cache = {}
        def expand(z):
            full = start.copy(); full[idx] = z
            return full
        def evaluate(z):
            if "z" not in cache or not np.array_equal(cache["z"], z):
                mu, jac = forward.model(expand(z))
                r, dr = syn.poisson_residual(mu[mask], observed[mask])
                cache.update(z=z.copy(), r=r, J=jac[:, mask].T[:, idx]*dr[:, None])
            return cache
        t0 = time.perf_counter()
        fit = least_squares(lambda z: evaluate(z)["r"], start[idx], jac=lambda z: evaluate(z)["J"],
            bounds=(lower[idx], upper[idx]), x_scale="jac", max_nfev=80,
            ftol=1e-7, xtol=1e-7, gtol=1e-6)
        theta = expand(fit.x)
        records.append({"theta":theta, "cost":float(fit.cost), "success":bool(fit.success),
            "status":int(fit.status), "nfev":int(fit.nfev), "start":k,
            "seconds":time.perf_counter()-t0,
            "bound_hits":int(np.sum(np.minimum(theta-lower, upper-theta)[idx]<1e-4))})
    best = min(records, key=lambda r:r["cost"])
    best["all_starts"] = [{k:v for k,v in r.items() if k != "theta"} for r in records]
    return best


def metrics(forward, theta, observed, expected, truth, mm):
    mu, _ = forward.model(theta)
    exact = forward.exact_intensity(theta)*np.exp(theta[-2])+np.exp(theta[-1])
    vals = {}
    for band in ["train", "validation", "test"]:
        r, _ = syn.poisson_residual(mu[mm[band]], observed[mm[band]])
        vals[band+"_deviance_mean"] = float(np.mean(r*r))
    vals["surface_relative_rmse"] = syn.shape_error(forward.family, theta, truth)
    vals["clean_test_intensity_relative_l2"] = float(np.linalg.norm((exact-expected)[mm["test"]])/np.linalg.norm(expected[mm["test"]]))
    return vals


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    snap_path = OUT / "preservation_snapshot.json"
    if snap_path.exists():
        before = json.loads(snap_path.read_text(encoding="utf-8"))
        assert all(sha(ROOT/p)==s for p,s in before.items()), "Prior evidence changed"
    else:
        before = protected(); dump(snap_path, before)
    dump(OUT / "experiment_config.json", {
        "date":"2026-10-12 Asia/Seoul", "calibration_seeds":[91000,91001,91002],
        "extra_lambdas":EXTRA_LAMBDAS, "noise_test_seeds_per_shape":5,
        "selection":"Minimum validation deviance among finite tested penalties including zero; bulk limit reported separately",
        "bulk_limit":"Six bulk/nuisance parameters; eight extra shape modes exactly zero",
        "data":"Same exact finite-mesh truth, q grid, photon level and original five paired noise draws",
        "reuse":"Seven historical calibration fits per shape and historical test fit when selected penalty exactly matches",
        "scope":"Supplementary pilot, no new photon levels or experimental cohort"})
    forward = syn.DifferentiableForward(syn.MeshFamily())
    mm = syn.masks(forward)
    old_cal = json.loads((PRIOR / "lambda_calibration.json").read_text(encoding="utf-8"))
    syn.prepare_coarse_library(forward)
    cal_rows, evaluated = [], []
    selections = []
    for model_no, (name, _) in enumerate(syn.targets()):
        source = np.load(PRIOR / f"{name}_zero_noise_seed-1.npz")
        truth, expected = source["truth"].copy(), source["expected"].copy()
        observed = np.random.default_rng(91000+model_no).poisson(expected).astype(float)
        starts = syn.initializations(forward, observed, mm["train"])
        candidates = []
        for old in [r for r in old_cal if r["model"]==name]:
            theta = np.asarray(old["theta"],float)
            vr, _ = syn.poisson_residual(forward.model(theta)[0][mm["validation"]], observed[mm["validation"]])
            value = float(np.mean(vr*vr))
            assert abs(value-old["validation_deviance_mean"])<1e-10
            row = {"model":name,"kind":"finite_penalty","lambda":old["lambda"],
                "validation_deviance_mean":value,"success":old["success"],
                "bound_hits":old["bound_hits"],"theta":theta,"reused":True}
            candidates.append(row); cal_rows.append(row)
        for lam in EXTRA_LAMBDAS:
            path = OUT / f"calibration_{name}_lambda{lam:g}.json"
            if path.exists():
                fit = json.loads(path.read_text(encoding="utf-8"))["fit"]
            else:
                fit = syn.fit(forward, observed, mm["train"], starts, lam)
                dump(path,{"model":name,"lambda":lam,"fit":fit})
            theta = np.asarray(fit["theta"],float)
            vr, _ = syn.poisson_residual(forward.model(theta)[0][mm["validation"]],observed[mm["validation"]])
            row = {"model":name,"kind":"finite_penalty","lambda":lam,
                "validation_deviance_mean":float(np.mean(vr*vr)),"success":fit["success"],
                "bound_hits":fit["bound_hits"],"theta":theta,"reused":False}
            assert row["success"], f"Calibration failed {name} {lam}"
            candidates.append(row); cal_rows.append(row)
            print(f"CAL {name} lambda {lam:g} val {row['validation_deviance_mean']:.10f}",flush=True)
        path = OUT / f"calibration_{name}_bulk_limit.json"
        if path.exists():
            fit = json.loads(path.read_text(encoding="utf-8"))["fit"]
        else:
            fit = bulk_limit(forward,observed,mm["train"],starts)
            dump(path,{"model":name,"kind":"bulk_limit","fit":fit})
        theta = np.asarray(fit["theta"],float)
        assert np.all(theta[4:-2]==0) and fit["success"]
        vr, _ = syn.poisson_residual(forward.model(theta)[0][mm["validation"]],observed[mm["validation"]])
        bulk_row={"model":name,"kind":"bulk_limit","lambda":None,
            "validation_deviance_mean":float(np.mean(vr*vr)),"success":fit["success"],
            "bound_hits":fit["bound_hits"],"theta":theta,"reused":False}
        cal_rows.append(bulk_row)
        chosen=min(candidates,key=lambda r:(r["validation_deviance_mean"],r["lambda"]))
        positive=min([r for r in candidates if r["lambda"]>0],key=lambda r:(r["validation_deviance_mean"],r["lambda"]))
        selections.append({"model":name,"selected_lambda":chosen["lambda"],
            "selected_validation_deviance":chosen["validation_deviance_mean"],
            "best_positive_lambda":positive["lambda"],"bulk_limit_validation_deviance":bulk_row["validation_deviance_mean"],
            "finite_endpoint_selected":chosen["lambda"]==max(EXTRA_LAMBDAS),
            "bulk_limit_better":bulk_row["validation_deviance_mean"]<chosen["validation_deviance_mean"],
            "test_used_for_selection":False,"truth_used_for_selection":False})
        dump(OUT / "selection.json", selections)
        dump(OUT / "calibration.json",cal_rows)
        print(f"SELECT {name} lambda {chosen['lambda']:g}; bulk limit {bulk_row['validation_deviance_mean']:.10f}",flush=True)
        for seed in range(5):
            paired = np.load(PRIOR/f"{name}_noise_seed{seed}.npz")
            y=paired["observed"].copy()
            assert np.array_equal(y,np.random.default_rng(11000+model_no*100+seed).poisson(expected))
            starts_seed=syn.initializations(forward,y,mm["train"])
            for condition, lam in [("selected_penalty",chosen["lambda"]),("positive_control",positive["lambda"]),("bulk_limit",None)]:
                path=OUT/f"{name}_{condition}_seed{seed}.json"
                if path.exists():
                    fitted=json.loads(path.read_text(encoding="utf-8"))["fit"]
                    reused=False
                elif condition=="bulk_limit":
                    fitted=bulk_limit(forward,y,mm["train"],starts_seed);reused=False
                else:
                    old_condition="noise" if lam==0 else "noise_tikhonov"
                    old_path=PRIOR/f"{name}_{old_condition}_seed{seed}.json"
                    old_fit=json.loads(old_path.read_text(encoding="utf-8"))
                    if old_fit["metrics"]["lambda"]==lam:
                        fitted=old_fit["fit"];reused=True
                    else:
                        fitted=syn.fit(forward,y,mm["train"],starts_seed,lam);reused=False
                theta=np.asarray(fitted["theta"],float)
                assert fitted["success"], f"Fit not converged: {name} {seed} {condition}"
                vals=metrics(forward,theta,y,expected,truth,mm)
                row={"model":name,"condition":condition,"seed":seed,"lambda":lam,
                    "success":fitted["success"],"bound_hits":fitted["bound_hits"],"reused":reused,**vals}
                dump(path,{"metrics":row,"fit":fitted,"truth":truth})
                evaluated.append(row)
                print(f"FIT {name} {condition} seed {seed} clean-test {vals['clean_test_intensity_relative_l2']:.6f}",flush=True)
        save_csv(OUT / "fit_metrics.csv",evaluated)
    table=[]
    for r in cal_rows:
        table.append({k:v for k,v in r.items() if k!="theta"})
    save_csv(OUT / "calibration_metrics.csv",table)
    groups=[]
    for name,_ in syn.targets():
        for condition in ["selected_penalty","positive_control","bulk_limit"]:
            rows=[r for r in evaluated if r["model"]==name and r["condition"]==condition]
            groups.append({"model":name,"condition":condition,"n":len(rows),"lambda":rows[0]["lambda"],
                **{k+"_median":float(np.median([r[k] for r in rows])) for k in ["surface_relative_rmse","clean_test_intensity_relative_l2","test_deviance_mean"]},
                "successes":sum(r["success"] for r in rows)})
    dump(OUT / "summary.json",{"groups":groups,"selection":selections,"calibration_candidates":len(cal_rows),
        "evaluated_rows":len(evaluated),"scope":"Three fixed synthetic models, same photon level, one independent calibration draw and five noise draws per model"})
    save_csv(OUT / "summary.csv",groups)
    assert all(sha(ROOT/p)==s for p,s in before.items())
    dump(OUT / "validation_audit.json",{"status":"complete","all_converged":True,
        "historical_calibration_validation_recomputed":21,"finite_selection_includes_zero":True,
        "selection_independent_of_truth_and_test":True,"noise_draws_identical_to_prior":True,
        "prior_hashes_preserved":len(before),"source_sha256":sha(__file__),
        "endpoint_caveat":[r for r in selections if r["finite_endpoint_selected"]]})
    print("PENALTY COMPLETE",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["run","summary"],default="run",nargs="?")
    args=ap.parse_args()
    if args.stage=="run":run()
    else:print((OUT/"summary.json").read_text(encoding="utf-8"))

"""Small, predetermined local-start sensitivity controls for supplementary evidence.

Previous fits, source data, masks, classes and production code remain read-only.
No global orientation search or test-based selection is performed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from revision_common import ROOT, dump, save_csv, sha, np

OUT = ROOT / 'output/differentiable_si_validation_2026-10-12/initial_sensitivity'
ID57 = [0, 69, 138, 208]
GROEL = [3, 4, 8, 57, 70, 74]
STARTS = ['reference', 'minus', 'plus']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    idout = ROOT / 'output/differentiable_real_2026-10-11'
    gout = ROOT / 'output/differentiable_groel_2026-10-11'
    sources = [ROOT / 'validation/revision_differentiable_real.py',
               ROOT / 'validation/revision_differentiable_groel.py',
               ROOT / 'validation/revision_differentiable_synthetic.py',
               ROOT / 'validation/revision_gpu_boundary_general.py']
    for folder, cases, pattern in [(idout, ID57, 'frame'), (gout, GROEL, 'pattern')]:
        sources += [folder / 'experiment_config.json', folder / 'lambda_selection.json',
                    folder / 'validation_manifest.json']
        sources += [folder / f'fits/{pattern}_{i:03d}.json' for i in cases]
    sources += [gout / 'detector_bands.npz', idout / 'detector_masks.npz',
                ROOT / 'validation_supporting_package/section5_groel/retrieval_symD7_curated76/groel_retrieval_best_with_frames.csv']
    config = {'date': '2026-10-12', 'ID57_cases': ID57, 'GroEL_cases': GROEL,
      'case_selection': 'Predetermined ID57 endpoint/evenly spaced indices; GroEL existing first/last noncalibration audit cases from each historical class. No selection by fit or test scores.',
      'starts': STARTS,
      'reference_reuse': 'Previously completed exact same procedure: nuisance fit -> bulk -> refinement at fixed selected lambda and zero-lambda/positive controls. Predictions and objectives independently recomputed here.',
      'ID57_perturbation': {'log_scale': float(np.log(1.05)), 'log_aspect_x': .035, 'log_aspect_y': -.025, 'radius_base_nm': 2., 'delta_rx': .025, 'delta_ry': -.025, 'delta_rz': .035},
      'GroEL_perturbation': {'log_scale_xyz': np.log([1.04, .97, 1.02]), 'delta_rotation_xyz_rad': [.025, -.020, .030], 'log_prism_radius': float(np.log(1.03)), 'log_prism_length': float(np.log(.97))},
      'perturbation_application': 'Signed perturbation to the original template geometry/local orientation, with all smooth modes initially zero; nuisance signal/background fitted anew on the unchanged training subset. Both start signs undergo the same nuisance/bulk/refinement stages.',
      'ID57_lambdas': sorted(set([0., read(idout / 'lambda_selection.json')['lambda']])),
      'GroEL_lambdas': sorted(set([0., read(gout / 'lambda_selection.json')['positive_control_lambda']])),
      'penalty_selection': 'Inherited from previous calibration. No new lambda choice using these cases.',
      'start_selection': 'Lowest training objective (training subsample log-MSE + fixed penalty), with deterministic reference/minus/plus tie break. Held-out pixels never select a start.',
      'scope': 'Local initialization sensitivity conditional on original class, template orientation and alignment. No global convergence, noise stability, parameter uncertainty or unique structure recovery claim.',
      'ID57_fit': {'grid': 384, 'quadrature': 4, 'stride': 4, 'max_nfev': 65},
      'GroEL_fit': {'quadrature': 2, 'stride': 2, 'max_nfev': 90},
      'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in sources}}
    old = read(OUT / 'experiment_config.json') if (OUT / 'experiment_config.json').exists() else None
    if old:
        assert old == json.loads(json.dumps(config, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))
    else:
        dump(OUT / 'experiment_config.json', config)
    verify_sources(config)
    print('PREPARED', {'ID57': ID57, 'GroEL': GROEL, 'new_starts': 20, 'new_optimization_stages': 80}, flush=True)


def verify_sources(config):
    for path, expected in config['source_hashes'].items():
        assert sha(ROOT / path) == expected, path


def result_path(dataset, i, label):
    folder = OUT / 'fits'
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f'{dataset}_{i:03d}_{label}.json'


def id57_cases():
    import revision_differentiable_real as r
    cfg, data, images, fullmask, seeds = r.context()
    local = {k: v[::4, ::4] for k, v in fullmask.items()}
    family = r.CoreShellFamily()
    fwd = r.forward(family, data['qaxis'][::4], 384)
    full = r.forward(family, data['qaxis'], 384)
    config = read(OUT / 'experiment_config.json')
    lambdas = config['ID57_lambdas']
    perturb = np.r_[np.log(1.05), .035, -.025, 2., .025, -.025, .035, np.zeros(6)]
    from scipy.optimize import nnls
    for i in ID57:
        family.rotation = data['rotation_matrices'][int(seeds[i]['orientation_index'])].astype(float)
        prior = read(r.OUT / f'fits/frame_{i:03d}.json')
        y = images[i, ::4, ::4]
        for label in STARTS:
            path = result_path('ID57', i, label)
            if path.exists() and path.with_suffix('.npz').exists():
                continue
            t = r.start_theta(seeds[i])
            sign = {'reference': 0., 'minus': -1., 'plus': 1.}[label]
            t = np.clip(t + sign * perturb, r.LOW + 1e-8, r.HIGH - 1e-8)
            original_start = t.copy()
            if label == 'reference':
                coarse = prior['coarse']; bulk = prior['bulk']
                fits = {str(lam): prior['fits'][str(lam)] for lam in lambdas}
            else:
                raw = np.abs(fwd.amplitude(t, False)[0]) ** 2
                norm = float(raw[local['train']].max())
                a, _ = nnls(np.c_[raw[local['train']] / norm, np.ones(local['train'].sum())], y[local['train']])
                t[-2:] = np.log([max(a[0] / norm, 1e-17), max(a[1], .001)])
                coarse = r.optimize(fwd, y, local['train'], t, [11, 12], 0., 65)
                bulk = r.optimize(fwd, y, local['train'], np.asarray(coarse['theta']), list(range(7)) + [11, 12], 0., 65)
                fits = {str(lam): r.optimize(fwd, y, local['train'], np.asarray(bulk['theta']), list(range(13)), lam, 65) for lam in lambdas}
            record = {'dataset': 'ID57', 'case_index': i, 'label': label, 'source_reused': label == 'reference',
                      'original_geometry_start': original_start, 'rotation': family.rotation, 'coarse': coarse, 'bulk': bulk, 'fits': fits}
            predictions = {}
            for stage, fit, lam in [('coarse', coarse, 0.), ('bulk', bulk, 0.)] + [('lambda_' + k, v, float(k)) for k, v in fits.items()]:
                theta = np.asarray(fit['theta'])
                pred = r.full_predict(full, theta)
                fit['metrics'] = {k: r.metric(pred, images[i], v) for k, v in fullmask.items() if k != 'roi'}
                trainpred = r.full_predict(fwd, theta)
                fit['train_subsample_log_mse'] = r.metric(trainpred, y, local['train'])['log_mse']
                fit['penalty_component'] = float(lam * np.sum((r.penalty() @ theta) ** 2))
                fit['training_objective_recomputed'] = fit['train_subsample_log_mse'] + fit['penalty_component']
                assert np.isclose(fit['training_objective_recomputed'], 2. * fit['cost'], rtol=1e-7, atol=1e-9)
                if stage.startswith('lambda_'):
                    predictions[stage] = pred[fullmask['roi']]
            np.savez_compressed(path.with_suffix('.npz'), **predictions)
            dump(path, record)
            print('DONE', 'ID57', i, label, [v['success'] for v in fits.values()], flush=True)
    import cupy as cp
    del full, fwd, images
    cp.get_default_memory_pool().free_all_blocks()


def groel_cases():
    import revision_differentiable_groel as r
    from scipy.optimize import nnls
    config = read(OUT / 'experiment_config.json')
    lambdas = config['GroEL_lambdas']
    for i in GROEL:
        row, img, masks, pix, subs, family = r.data_for(i)
        train = subs['train'] & (pix[:, 0] % 2 == 0) & (pix[:, 1] % 2 == 0)
        fwd = r.BoundaryForward(family, pix[train]); full = r.BoundaryForward(family, pix)
        y = img[masks['roi']]
        prior = read(r.OUT / f'fits/pattern_{i:03d}.json')
        perturb = np.zeros(16)
        perturb[:3] = np.log([1.04, .97, 1.02]); perturb[3:6] = [.025, -.020, .030]
        if family.prism.any():
            perturb[10:12] = np.log([1.03, .97])
        active = r.BULK + [6, 7, 8, 9] + ([10, 11] if family.prism.any() else [])
        for label in STARTS:
            path = result_path('GroEL', i, label)
            if path.exists() and path.with_suffix('.npz').exists():
                continue
            sign = {'reference': 0., 'minus': -1., 'plus': 1.}[label]
            t = sign * perturb; t[-2] = -20.; t[-1] = 0.
            t = np.clip(t, r.LOW + 1e-8, r.HIGH - 1e-8)
            original_start = t.copy()
            if label == 'reference':
                coarse = prior['coarse']; bulk = prior['bulk']
                fits = {str(lam): prior['fits'][str(lam)] for lam in lambdas}
            else:
                raw = np.abs(fwd.amplitude(t, False)[0]) ** 2; norm = raw.max()
                a, _ = nnls(np.c_[raw / norm, np.ones(len(raw))], y[train])
                t[-2] = np.log(max(a[0] / norm, 1e-14)); t[-1] = np.log(max(a[1], 1e-4))
                coarse = r.optimize(fwd, y[train], t, [14, 15], budget=90)
                bulk = r.optimize(fwd, y[train], np.asarray(coarse['theta']), r.BULK, budget=90)
                fits = {str(lam): r.optimize(fwd, y[train], np.asarray(bulk['theta']), active, lam, 90) for lam in lambdas}
            record = {'dataset': 'GroEL', 'case_index': i, 'source_frame_index': int(row['source_frame_index']),
                      'label': label, 'model_name': row['model_name'], 'source_reused': label == 'reference',
                      'original_geometry_start': original_start, 'rotation': family.rotation, 'coarse': coarse, 'bulk': bulk, 'fits': fits}
            predictions = {}
            for stage, fit, lam in [('coarse', coarse, 0.), ('bulk', bulk, 0.)] + [('lambda_' + k, v, float(k)) for k, v in fits.items()]:
                theta = np.asarray(fit['theta']); pred = full.predict(theta)
                fit['metrics'] = r.metrics(pred, y, subs)
                residual = np.log1p(fwd.predict(theta)) - np.log1p(y[train])
                fit['train_subsample_log_mse'] = float(np.mean(residual ** 2))
                fit['penalty_component'] = float(lam * np.sum((r.Lmatrix(family.prism.any()) @ theta) ** 2))
                fit['training_objective_recomputed'] = fit['train_subsample_log_mse'] + fit['penalty_component']
                assert np.isclose(fit['training_objective_recomputed'], 2. * fit['cost'], rtol=1e-7, atol=1e-9)
                if stage.startswith('lambda_'):
                    predictions[stage] = pred
            np.savez_compressed(path.with_suffix('.npz'), **predictions); dump(path, record)
            print('DONE', 'GroEL', i, label, [v['success'] for v in fits.values()], flush=True)
        del fwd, full
        r.cp.get_default_memory_pool().free_all_blocks()


def summarize():
    config = read(OUT / 'experiment_config.json'); verify_sources(config)
    rows = []; parameters = []; comparisons = []; qa = []
    for dataset, cases, key in [('ID57', ID57, 'ID57_lambdas'), ('GroEL', GROEL, 'GroEL_lambdas')]:
        if dataset == 'ID57':
            import revision_differentiable_real as mod
            active_geom = list(range(11)); primary = max(config[key])
            _, _, target_images, target_masks, _ = mod.context()
        else:
            import revision_differentiable_groel as mod
            active_geom = list(range(14)); primary = 0.
        for i in cases:
            if dataset == 'ID57':
                target = target_images[i][target_masks['roi']]
                target_subs = {k: target_masks[k][target_masks['roi']] for k in ['train', 'validation', 'test']}
            else:
                _, img, masks, _, target_subs, _ = mod.data_for(i)
                target = img[masks['roi']]
            records = {s: read(result_path(dataset, i, s)) for s in STARTS}
            for s, rec in records.items():
                assert rec['dataset'] == dataset and rec['case_index'] == i and rec['label'] == s
                for stage, fit, lam in [('coarse', rec['coarse'], 0.), ('bulk', rec['bulk'], 0.)] + [('lambda_' + k, v, float(k)) for k, v in rec['fits'].items()]:
                    theta = np.asarray(fit['theta']); assert np.isfinite(theta).all()
                    assert ((theta >= mod.LOW - 1e-9) & (theta <= mod.HIGH + 1e-9)).all()
                    assert abs(fit['training_objective_recomputed'] - 2. * fit['cost']) < 1e-7
                    row = {'dataset': dataset, 'case_index': i, 'start': s, 'stage': stage, 'lambda': lam,
                           'success': fit['success'], 'nfev': fit['nfev'], 'bound_hits': fit['bound_hits'],
                           'train_subsample_log_mse': fit['train_subsample_log_mse'], 'penalty_component': fit['penalty_component'],
                           'training_objective': fit['training_objective_recomputed'],
                           'train_log_mse': fit['metrics']['train']['log_mse'], 'validation_log_mse': fit['metrics']['validation']['log_mse'],
                           'test_log_mse': fit['metrics']['test']['log_mse'], 'reference_reused': rec['source_reused']}
                    rows.append(row)
                    for j, name in enumerate(mod.NAMES):
                        parameters.append({'dataset': dataset, 'case_index': i, 'start': s, 'stage': stage, 'parameter': name, 'value': theta[j]})
            # Original source controls are independently re-evaluated, never refitted.
            prefix = 'frame' if dataset == 'ID57' else 'pattern'
            prior = read(mod.OUT / f'fits/{prefix}_{i:03d}.json')
            reference = records['reference']
            for lam in config[key]:
                ref = reference['fits'][str(lam)]
                for split in ['train', 'validation', 'test']:
                    assert np.isclose(ref['metrics'][split]['log_mse'], prior['fits'][str(lam)]['metrics'][split]['log_mse'], rtol=1e-8, atol=1e-10)
                fits = [records[s]['fits'][str(lam)] for s in STARTS]
                chosen_index = min(range(3), key=lambda j: fits[j]['training_objective_recomputed'])
                chosen = fits[chosen_index]; theta = np.asarray([v['theta'] for v in fits])
                pred = [np.load(result_path(dataset, i, s).with_suffix('.npz'))['lambda_' + str(lam)] for s in STARTS]
                for p, fit in zip(pred, fits):
                    assert p.shape == target.shape and np.isfinite(p).all() and (p >= 0).all()
                    residual2 = (np.log1p(p) - np.log1p(target)) ** 2
                    for split in ['train', 'validation', 'test']:
                        independently_recomputed = float(np.mean(residual2[target_subs[split]]))
                        assert np.isclose(independently_recomputed, fit['metrics'][split]['log_mse'], rtol=1e-9, atol=1e-11)
                scaled = (theta[:, active_geom] - theta[0, active_geom]) / (mod.HIGH - mod.LOW)[active_geom]
                comparisons.append({'dataset': dataset, 'case_index': i, 'lambda': lam, 'primary': lam == primary,
                    'training_selected_start': STARTS[chosen_index], 'training_selected_start_converged': chosen['success'],
                    'all_three_converged': all(v['success'] for v in fits),
                    'reference_test_log_mse': fits[0]['metrics']['test']['log_mse'],
                    'minus_test_log_mse': fits[1]['metrics']['test']['log_mse'], 'plus_test_log_mse': fits[2]['metrics']['test']['log_mse'],
                    'training_selected_test_log_mse': chosen['metrics']['test']['log_mse'],
                    'selected_vs_reference_test_reduction_pct': 100. * (1. - chosen['metrics']['test']['log_mse'] / fits[0]['metrics']['test']['log_mse']),
                    'max_test_log_mse_span': max(v['metrics']['test']['log_mse'] for v in fits) - min(v['metrics']['test']['log_mse'] for v in fits),
                    'max_validation_log_mse_span': max(v['metrics']['validation']['log_mse'] for v in fits) - min(v['metrics']['validation']['log_mse'] for v in fits),
                    'max_roi_log_prediction_rms_to_reference': max(float(np.sqrt(np.mean((np.log1p(p) - np.log1p(pred[0])) ** 2))) for p in pred),
                    'max_geometry_bound_span_normalized_l2_to_reference': float(np.linalg.norm(scaled, axis=1).max()),
                    'max_abs_geometry_difference_to_reference': float(np.max(np.abs(theta[:, active_geom] - theta[0, active_geom]))),
                    'maximum_bound_hits': max(v['bound_hits'] for v in fits)})
                qa.append({'dataset': dataset, 'case_index': i, 'lambda': lam, 'reference_metrics_reproduced': True,
                           'all_start_metrics_independently_recomputed_from_saved_predictions_and_raw_targets': True,
                           'all_predictions_finite_nonnegative': all(np.isfinite(p).all() and (p >= 0).all() for p in pred)})
        if dataset == 'ID57':
            del target_images, target_masks
    save_csv(OUT / 'fit_metrics.csv', rows); save_csv(OUT / 'parameter_metrics.csv', parameters)
    save_csv(OUT / 'start_comparisons.csv', comparisons)
    aggregates = []
    for dataset in ['ID57', 'GroEL']:
        for lam in config[dataset + '_lambdas']:
            sub = [r for r in comparisons if r['dataset'] == dataset and r['lambda'] == lam]
            ref = np.mean([r['reference_test_log_mse'] for r in sub]); sel = np.mean([r['training_selected_test_log_mse'] for r in sub])
            aggregates.append({'dataset': dataset, 'lambda': lam, 'n_cases': len(sub),
               'all_converged': all(r['all_three_converged'] for r in sub),
               'all_training_selected_starts_converged': all(r['training_selected_start_converged'] for r in sub),
               'reference_mean_test_log_mse': ref, 'training_selected_mean_test_log_mse': sel,
               'selected_vs_reference_test_reduction_pct': 100. * (1. - sel / ref),
               'n_training_selected_test_improved': sum(r['training_selected_test_log_mse'] < r['reference_test_log_mse'] - 1e-10 for r in sub),
               'n_training_selected_test_worsened': sum(r['training_selected_test_log_mse'] > r['reference_test_log_mse'] + 1e-10 for r in sub),
               'maximum_test_log_mse_span': max(r['max_test_log_mse_span'] for r in sub),
               'maximum_roi_log_prediction_rms_to_reference': max(r['max_roi_log_prediction_rms_to_reference'] for r in sub),
               'maximum_geometry_bound_span_normalized_l2_to_reference': max(r['max_geometry_bound_span_normalized_l2_to_reference'] for r in sub),
               'selected_start_counts': {s: sum(r['training_selected_start'] == s for r in sub) for s in STARTS}})
    dump(OUT / 'summary.json', {'aggregates': aggregates, 'selection_test_based': False,
         'interpretation': 'Small local-start sensitivity controls only. ID57 shows local initialization dependence: training-based selection worsens test prediction for case 69, while case 208 improves substantially; the four-case mean is not a general performance claim. GroEL predictions are similar under the tested local perturbations, while parameters can differ. No confidence intervals or identifiability claim.',
         'nonconvergence': [{'dataset': r['dataset'], 'case_index': r['case_index'], 'start': r['start'], 'stage': r['stage'], 'nfev': r['nfev']} for r in rows if not r['success']], 'cases': comparisons})
    dump(OUT / 'validation_audit.json', {'case_count': len(ID57) + len(GROEL), 'start_count': 30,
         'new_start_count': 20, 'fit_metric_rows': len(rows), 'comparison_rows': len(comparisons), 'checks': qa,
         'parameter_bounds_checked': True, 'training_objective_independently_recomputed': True,
         'test_based_start_selection': False, 'source_hashes_unchanged': True,
         'all_fits_converged': all(r['success'] for r in rows)})
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'validation_manifest.json')
    dump(OUT / 'validation_manifest.json', {'validation_status': 'complete', 'sources': config['source_hashes'],
         'new_source': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
         'outputs': {str(p.relative_to(ROOT)): sha(p) for p in files},
         'scope': config['scope']})
    print('SUMMARY', json.dumps(aggregates), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['prepare', 'id57', 'groel', 'all', 'summarize'])
    a = ap.parse_args()
    if a.stage == 'prepare': prepare()
    elif a.stage == 'summarize': summarize()
    else:
        verify_sources(read(OUT / 'experiment_config.json'))
        if a.stage in ['id57', 'all']: id57_cases()
        if a.stage in ['groel', 'all']: groel_cases()

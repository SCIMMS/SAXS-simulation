"""Audit retained scientific evidence without importing research/GPU modules."""
from pathlib import Path
import argparse, ast, csv, hashlib, json, math, statistics, zipfile

PACKAGE=Path(__file__).resolve().parents[1]
REPO=PACKAGE.parents[1]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-10):
    assert math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol),(a,b)
def study(name):return PACKAGE/'data/output'/name

def verify(numerical=False):
    manifest=json.loads((PACKAGE/'MANIFEST.json').read_text(encoding='utf-8'))
    count=0;total=0
    for r in manifest['files']:
        p=REPO/r['path'];assert p.is_file(),p
        assert p.stat().st_size==r['bytes'],str(p)
        assert sha(p)==r['sha256'],str(p)
        count+=1;total+=r['bytes']
    source_records=json.loads((PACKAGE/'provenance/source_records.json').read_text(encoding='utf-8'))
    for r in source_records:assert sha(REPO/r['path'])==r['published_sha256'],r['path']
    modules=list((PACKAGE/'scripts').rglob('*.py'))
    for p in modules:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    assets=rows(PACKAGE/'provenance/asset_index.csv')
    assert len(assets)==50
    for r in assets:
        folder=REPO/r['repository_relative_folder'];assert (folder/'README.md').is_file()
        for s in r['source_paths'].split(';'):
            if s:assert (REPO/s).exists(),s
        if 'Figure' in r['item']:
            pp=list((folder/'images').glob('*.png'));assert len(pp)==1
        else:pp=[folder/'data/table.csv']
        assert sha(pp[0])==r['image_or_table_sha256']

    bank=study('id57_qz5_rounding_2026-10-16')
    candidates=rows(bank/'candidate_parameters.csv');assert len(candidates)==8400
    selected=rows(bank/'selection_results.csv');assert len(selected)==209*3*3
    lookup={(r['fold'],int(r['frame']),r['family']):r for r in selected}
    assert len(lookup)==len(selected)
    deltas={}
    summary=json.loads((bank/'selection_summary.json').read_text(encoding='utf-8'))['results']
    for fold in ['full','even_to_odd','odd_to_even']:
        ds=[float(lookup[fold,i,'rounded']['test_logcorr'])-float(lookup[fold,i,'variable_size_sharp']['test_logcorr']) for i in range(209)]
        deltas[fold]=ds;close(statistics.mean(ds),summary[fold]['mean_rounding_over_size_delta'])
        for family in ['fixed_sharp','variable_size_sharp','rounded']:
            close(statistics.mean(float(lookup[fold,i,family]['test_logcorr']) for i in range(209)),summary[fold]['mean_scores'][family])
    paired=[(x+y)/2 for x,y in zip(deltas['even_to_odd'],deltas['odd_to_even'])]
    close(statistics.mean(paired),0.046175827087112986)
    assert sum(x>0 for x in paired)==187
    exact=rows(bank/'selected_exact_reference_fits.csv');assert len(exact)==len(selected)
    ex={(r['fold'],int(r['frame']),r['family']):float(r['exact_test_logcorr']) for r in exact}
    exact_paired=[statistics.mean(ex[fold,i,'rounded']-ex[fold,i,'variable_size_sharp'] for fold in ['even_to_odd','odd_to_even']) for i in range(209)]
    close(statistics.mean(exact_paired),0.0461777502,1e-9)
    assert len({int(r['candidate']) for r in exact})==701
    baseline=rows(study('revision_remaining_2026-10-09')/'02_baseline/id57_baseline_results.csv')
    for mode in ['full','heldout','heldout_reverse']:
        rs=[r for r in baseline if r['mode']==mode];assert len(rs)==209
    aspect=rows(study('qz5_aspect_sensitivity_2026-10-16')/'paired_results.csv')
    for label,bounds,median in [('narrow',180,10.62),('medium',142,10.51),('wide',66,10.80)]:
        rs=[r for r in aspect if r['range_label']==label];assert len(rs)==209
        assert sum(r['aspect_bound']=='True' for r in rs)==bounds
        assert abs(statistics.median(float(r['radius_nm']) for r in rs)-median)<.006
    ab=rows(study('groel_ab_matched_2026-10-15')/'paired_results.csv')
    assert len(ab)==76*3*2
    for fold,expected in [('full',41),('even_to_odd',27),('odd_to_even',28)]:
        by={(int(r['pattern_index']),r['model']):r for r in ab if r['fold']==fold}
        assert len(by)==152
        assert sum(float(by[i,'A']['test_rms'])<float(by[i,'B']['test_rms']) for i in range(76))==expected
    for name,expected in [('differentiable_real_2026-10-11',.1),('differentiable_groel_2026-10-11',0.)]:
        selected_lambda=json.loads((study(name)/'lambda_selection.json').read_text(encoding='utf-8'))
        close(selected_lambda['lambda'],expected)
    close(json.loads((study('differentiable_groel_2026-10-11')/'lambda_selection.json').read_text(encoding='utf-8'))['positive_control_lambda'],1e-4)
    if numerical:
        import numpy as np
        families={'fixed_sharp':np.arange(240)*5+2,'variable_size_sharp':np.arange(1200),'rounded':np.arange(8400)}
        rank_checks=0;max_score_gap=0.
        for fold in ['full','even_to_odd','odd_to_even']:
            scores=np.load(bank/f'scores_{fold}.npy',mmap_mode='r')
            assert scores.shape==(209,8400) and scores.dtype==np.float64
            assert np.isfinite(scores).all()
            for i in range(209):
                for family,ids in families.items():
                    r=lookup[fold,i,family];chosen=int(r['candidate'])
                    assert chosen in ids
                    # The original CPU finalist stage may resolve roundoff ties.
                    gap=float(np.max(scores[i,ids])-scores[i,chosen]);assert gap<1e-8,(fold,i,family,gap)
                    close(scores[i,chosen],r['fit_logcorr'],1e-8)
                    max_score_gap=max(max_score_gap,gap);rank_checks+=1
        npz_count=0
        for p in (PACKAGE/'data').rglob('*.npz'):
            with zipfile.ZipFile(p) as z:assert z.testzip() is None,str(p)
            npz_count+=1
        with np.load(PACKAGE/'data/inputs/id57_geometry_fixture.npz',allow_pickle=False) as z:
            assert z['rotation_matrices'].shape==(240,3,3)
            assert np.isfinite(z['rotation_matrices']).all()
            mask=z['valid_mask'].astype(bool);assert mask.shape==(401,401)
            yy,xx=np.indices(mask.shape);radius=np.hypot(xx-200,yy-200)
            assert (mask&(radius>=45)&(radius<=190)).sum()==91567
        numerical_result=dict(selected_candidate_rank_checks=rank_checks,maximum_training_score_gap=max_score_gap,npz_archives_crc_checked=npz_count)
    else:numerical_result='not requested'
    return dict(status='pass',manifest_files_verified=count,manifest_bytes_verified=total,python_modules_parsed=len(modules),assets_verified=len(assets),id57_frames=209,rounded_candidates=8400,selected_exact_geometries=701,paired_withheld_gain=statistics.mean(paired),exact_reference_paired_gain=statistics.mean(exact_paired),numerical=numerical_result,scope='Saved evidence audit only; no GPU simulation, fitting or timing rerun.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--numerical',action='store_true')
    args=parser.parse_args();print(json.dumps(verify(args.numerical),indent=2))

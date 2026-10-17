"""Sharp-like start and independently refined target-mesh checks."""
from revision_differentiable_synthetic import *

def run():
    family=MeshFamily();forward=DifferentiableForward(family);mm=masks(forward)
    prepare_coarse_library(forward)
    source=np.load(OUT/'rounded_box_zero_noise_seed-1.npz')
    truth=source['truth'].copy();expected=source['expected']
    sharp_start=initializations(forward,expected,mm['train'])[0]
    forced=fit(forward,expected,mm['train'],[sharp_start])
    sharp={'initial_exponent':float(np.exp(sharp_start[2])),
      'recovered_exponent':float(np.exp(forced['theta'][2])),
      'target_exponent':float(np.exp(truth[2])),
      'surface_relative_rmse':shape_error(family,forced['theta'],truth),'fit':forced,
      'scope':'Exponent 12 is box-like with finite rounding; it is not an exact sharp cube.'}
    dump(OUT/'box_like_initialization_check.json',sharp)
    print('BOX-LIKE START',sharp['initial_exponent'],'->',sharp['recovered_exponent'],flush=True)
    fine=DifferentiableForward(MeshFamily(nlat=32,nlon=96))
    exact_fine=fine.exact_intensity(truth)
    truth[-2]=np.log(20000./exact_fine[mm['valid']].sum())
    expected=exact_fine*np.exp(truth[-2])+np.exp(truth[-1])
    exact_coarse=forward.exact_intensity(truth)*np.exp(truth[-2])+np.exp(truth[-1])
    discrepancy=np.linalg.norm((exact_coarse-expected)[mm['valid']])/np.linalg.norm(expected[mm['valid']])
    noisy=np.random.default_rng(92000).poisson(expected).astype(float)
    rows=[]
    for condition,observed,lam in [('zero_noise',expected,0.),('noise',noisy,0.),('noise_tikhonov',noisy,1000.)]:
        starts=initializations(forward,observed,mm['train'])
        fitted=fit(forward,observed,mm['train'],starts,lam)
        theta=fitted['theta']
        exact=forward.exact_intensity(theta)*np.exp(theta[-2])+np.exp(theta[-1])
        test_error=np.linalg.norm((exact-expected)[mm['test']])/np.linalg.norm(expected[mm['test']])
        row={'condition':condition,'lambda':lam,'surface_relative_rmse':shape_error(family,theta,truth),
          'clean_test_intensity_relative_l2':float(test_error),'success':fitted['success'],
          'exponent':float(np.exp(theta[2]))}
        rows.append(row)
        dump(OUT/f'fine_target_{condition}.json',{'metrics':row,'truth':truth,'fit':fitted})
        print('FINE TARGET',condition,'shape',row['surface_relative_rmse'],'test',test_error,flush=True)
    robust={'target_faces':len(fine.family.faces),'fit_faces':len(family.faces),
      'coarse_fine_reference_discrepancy_relative_l2':float(discrepancy),'noise_seed':92000,
      'lambda_source':'Fixed at 1000 from the primary rounded-box calibration; not tuned on this target',
      'groups':rows,'caveat':'One shape and one noise draw; supplementary numerical robustness check only.'}
    dump(OUT/'fine_target_robustness.json',robust)
    # Check the full intensity -> Poisson residual -> Tikhonov Jacobian chain.
    data=np.load(OUT/'deformed_rounded_box_noise_seed0.npz');theta=data['truth'].copy();y=data['observed']
    L=regularizer();lam=10.;mask=mm['train']
    def objective(x):
        mu,j=forward.model(x);r,dr=poisson_residual(mu[mask],y[mask])
        return np.r_[r,np.sqrt(lam)*(L@x)],np.vstack([j[:,mask].T*dr[:,None],np.sqrt(lam)*L])
    _,J=objective(theta);rng=np.random.default_rng(333);errs=[]
    for _ in range(3):
        direction=rng.normal(size=len(theta));direction/=np.linalg.norm(direction);h=1e-7
        numerical=(objective(theta+h*direction)[0]-objective(theta-h*direction)[0])/(2*h)
        analytical=J@direction
        errs.append(float(np.linalg.norm(numerical-analytical)/np.linalg.norm(numerical)))
    assert max(errs)<.001
    edges=np.sort(np.concatenate([family.faces[:,[0,1]],family.faces[:,[1,2]],family.faces[:,[2,0]]]),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True);assert np.all(counts==2)
    dump(OUT/'full_objective_gradient_qa.json',{'directional_relative_l2_errors':errs,
      'directions':3,'lambda':lam,'watertight_edges':True,'gate':.001})

if __name__=='__main__':run()

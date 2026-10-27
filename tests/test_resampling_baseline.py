"""Regression checks for runtime baselines, coordinate conversion and diagnostics."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from astropy.stats import mad_std
from test_broad_width_mapping import Gmodel, map_params, helpers

ROOT = Path(__file__).resolve().parents[1] / 'xgfit'
ns = dict(np=np, map_params=map_params, minimize=minimize, idx=helpers.idx)
for path, names in [(ROOT/'subroutines.py', {'fill_snr'}),
                    (ROOT/'xgfit/resampling.py', {'synchronize_tied_centers',
                     'fit_resampled_baseline', 'refresh_fit_diagnostics',
                     'fill_correlation_diagnostics'})]:
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    ns['mad_std'] = mad_std
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)


class ResamplingBaselineTest(unittest.TestCase):
    def test_final_adoption_refreshes_baselines_and_diagnostics(self):
        tree = ast.parse((ROOT/'xgfit/resampling.py').read_text())
        fill = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == 'fill_df_resampled')
        fill.decorator_list = []
        fill.args.defaults = []  # Supply statistics explicitly, without outer self.
        names = ['F21','F22','V21','S21','S22']
        names_1G = ['F1','V1','S1']
        rng = np.random.default_rng(656)
        samples = rng.normal([.2,1.3,-2.5,12.,37.], [.01,.02,.1,.2,.3], (100,5))
        one = rng.normal([1.5,-1.6,20.], [.01,.1,.2], (100,3))
        xx = np.linspace(-100,100,81)
        frame = pd.DataFrame([dict(F1=1.,V1=0.,S1=15.,B1=0.,B2=0.,
                                   V21=30.,V22=30.,N2=99.,SNR2=-1.)])
        def gauss(x,a,v,s):
            return a*np.exp(-.5*((x-v)/s)**2)
        def residuals(G):
            row = frame.iloc[0]
            if G==1:
                model = gauss(xx,row.A1,row.V1,row.S1)+row.B1
            else:
                model = gauss(xx,row.A21,row.V21,row.S21)+gauss(xx,row.A22,row.V22,row.S22)+row.B2
            return fitter.y-model
        fitter = SimpleNamespace(df=frame,df_params=pd.DataFrame([{}]),
                    truth_from_resampling=True,gmodel=SimpleNamespace(names_param=names,x=xx),
                    y=gauss(xx,.02,-2.,25.)+.01,e_y=np.full(len(xx),.002),x=xx,
                    get_residuals=residuals)
        env = dict(ns, self=fitter, names_1G=names_1G, G=2,
                   resampled_1G=one, fitB1='BB', BBs=np.linspace(.009,.011,100),
                   sort_outliers=lambda x: np.asarray(x), gauss=gauss,
                   gmodel_resample_1G=SimpleNamespace(names_param=names_1G),
                   trim_outlier_mahalanobis=lambda x,y:(x,y),spearmanr=spearmanr,
                   compute_aic_bic_aicc=lambda ll,k,n:(-2*ll+2*k,-2*ll+k*np.log(n),0.))
        exec(compile(ast.Module(body=[fill],type_ignores=[]),'<adoption-test>','exec'),env)
        env['fill_df_resampled'](samples,'MEDIAN')
        self.assertAlmostEqual(frame.loc[0,'B1'],.01)
        self.assertAlmostEqual(frame.loc[0,'B2'],.01)
        self.assertEqual(frame.loc[0,'V21'],frame.loc[0,'V22'])
        expected_noise=mad_std(residuals(2))
        self.assertAlmostEqual(frame.loc[0,'N2'],expected_noise)
        self.assertAlmostEqual(frame.loc[0,'SNR2'],(frame.loc[0,'A21']+frame.loc[0,'A22'])/expected_noise)
        self.assertAlmostEqual(fitter.df_params.loc[0,'SNR2'],frame.loc[0,'SNR2'])

    def test_runtime_baseline_matches_rebuilt_likelihood(self):
        variants = [
            (['F1','V1','S1'], 'make_log_prob_1G_B1x'),
            (['F21','F22','V21','S21','S22'], 'make_log_prob_2G_V22xB2x'),
            (['F21','F22','V21','V22','S21','S22'], 'make_log_prob_2G_B2x'),
            (['F31','F32','F33','V31','S31','S32','S33'], 'make_log_prob_3G_V32xV33xB3x'),
        ]
        xx=np.linspace(-80,80,81)
        for names, factory in variants:
            with self.subTest(factory=factory):
                bounds={n: ([0,5] if n.startswith('F') else [-10,10] if n.startswith('V')
                            else [.5,20] if n=='S31' else [2,20] if n in ('S1','S21','S32')
                            else [2,100]) for n in names}
                model=Gmodel(xx,np.zeros(len(xx)),np.full(len(xx),.01),names,bounds)
                model.BB=.003
                _, objective=getattr(model,factory)()
                u=np.full(len(names),.4)
                before=objective(u,model.y)
                model.BB=.02
                _, rebuilt=getattr(model,factory)()
                self.assertAlmostEqual(objective(u,model.y,.02),rebuilt(u,model.y),places=9)
                self.assertNotAlmostEqual(before,objective(u,model.y,.02))
                self.assertAlmostEqual(before,objective(u,model.y),places=9)

    def test_free_baseline_fit_recovers_physical_value(self):
        xx=np.linspace(-100,100,101)
        true=np.array([2.,3.,12.,.017])
        yy=true[0]/(true[2]*np.sqrt(2*np.pi))*np.exp(-.5*((xx-true[1])/true[2])**2)+true[3]
        model=Gmodel(xx,yy,np.full(len(xx),.002),['F1','V1','S1','B1'],
                     {'F1':[0,5], 'V1':[-20,20], 'S1':[1,40], 'B1':[-.05,.05]})
        _, model.log_prob_resample=model.make_log_prob_1G()
        fitted=ns['fit_resampled_baseline'](model,np.array([1.7,1.,10.,.01]),yy,'Nelder-Mead')
        self.assertAlmostEqual(fitted,true[3],places=5)
        # A numeric slice of a mixed result row retains object dtype in pandas.
        row = pd.Series({'Name':'TEST','F1':np.float64(1.7),'V1':np.float64(1.),
                         'S1':np.float64(10.),'B1':np.float64(.01)})
        object_guess = row[['F1','V1','S1','B1']]
        self.assertEqual(object_guess.dtype, object)
        mapped = map_params(object_guess, model, 'x->u')
        self.assertEqual(mapped.dtype, np.float64)
        np.testing.assert_allclose(map_params(mapped, model, 'u->x'),
                                   object_guess.to_numpy(dtype=np.float64))
        fitted = ns['fit_resampled_baseline'](model,object_guess,yy,'Nelder-Mead')
        self.assertAlmostEqual(fitted,true[3],places=5)
        real=ns['minimize']
        try:
            ns['minimize']=lambda *a,**kw: SimpleNamespace(success=False,fun=1.)
            with self.assertRaisesRegex(RuntimeError,'failed to converge'):
                ns['fit_resampled_baseline'](model,true,yy,'Nelder-Mead')
        finally:
            ns['minimize']=real

    def test_diagnostics_use_final_amplitudes_and_tied_centers(self):
        frame=pd.DataFrame([dict(V1=0.,V21=-2.,V22=30.,A1=.03,A21=.006,A22=.014,N2=99.,SNR2=-1.)])
        residual=np.array([-.002,-.001,0.,.001,.002])
        def get_residuals(G):
            if G==2:
                self.assertEqual(frame.loc[0,'V21'],frame.loc[0,'V22'])
            return residual
        fitter=SimpleNamespace(df=frame,get_residuals=get_residuals)
        ns['refresh_fit_diagnostics'](fitter,['V21'],2)
        noise=mad_std(residual)
        self.assertAlmostEqual(frame.loc[0,'N2'],noise)
        self.assertAlmostEqual(frame.loc[0,'SNR2'],.020/noise)
        self.assertAlmostEqual(frame.loc[0,'SNR21'],.006/noise)
        self.assertAlmostEqual(frame.loc[0,'SNR1'],.03/noise)


if __name__=='__main__':
    unittest.main()

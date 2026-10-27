"""Final-only plotting, failure checkpoints, and display-only chain thinning."""
import ast
import gc
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.config import FitConfig, PipelineConfig
from dataclasses import replace


def load_class(path, name, namespace):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
    module=ast.parse('from __future__ import annotations')
    module.body.append(node)
    exec(compile(module,str(path),'exec'),namespace)
    return namespace[name]


class PlotPolicyTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.plots=[]
        calls=self.plots
        class Plotter:
            def __init__(self,path,name,suffix,disp,df,params,model,sampler,resampled,*args):
                self.df=df.copy();self.model=model;self.sampler=sampler
                self.resampled=resampled;self.savename_atlas=path/'atlas.png'
                self.savename_corner_emcee=path/'corner.png'
                self.savename_corner_resample=path/'resample_corner.png'
            def makeplot_atlas(self,G):
                calls.append(self)
                self.savename_atlas.write_text(str(G))
            def cleanup(self):pass
        cls=load_class(ROOT/'xgfit/xgfit/atlas.py','AtlasMixin',
                       {'gc':gc,'plt':SimpleNamespace(close=lambda *a:None),'Plotter':Plotter})
        self.fitter=cls()
        for k,v in dict(path_plot=self.root,name_cube='TEST',suffix='_',list_disp=[],list_NHI_=None,
                        df=pd.DataFrame([dict(V21=1.,V22=1.)]),df_params=pd.DataFrame([dict(stage='MCMC')]),
                        gmodel=SimpleNamespace(names_param=['S21']),gmodel_1G=SimpleNamespace(names_param=['S1']),
                        sampler=object(),resampled=None,burnin=20,thin=1).items():
            setattr(self.fitter,k,v)

    def test_intermediate_disabled_but_final_and_error_saved(self):
        f=self.fitter
        f.make_atlas(intermediate=True)
        self.assertFalse(self.plots)
        (self.root/'corner.png').write_text('old run')
        f.make_atlas()
        self.assertEqual(len(self.plots),1)
        self.assertFalse((self.root/'corner.png').exists())
        self.assertEqual(self.plots[0].walk_plot_max_steps,1000)
        f.intermediate_plots=True
        f.make_atlas(intermediate=True)
        self.assertEqual(len(self.plots),2)

    def test_error_uses_last_completed_parameters_and_retains_exception(self):
        f=self.fitter
        f.checkpoint_plot_state(f.gmodel)
        saved_sampler=f.sampler
        f.df.loc[0,'V21']=999.
        f.sampler=None
        f.make_error_atlas()
        self.assertEqual(self.plots[-1].df.loc[0,'V21'],1.)
        self.assertIs(self.plots[-1].sampler,saved_sampler)
        self.assertEqual(f.df.loc[0,'V21'],999.)
        self.assertIsNone(f.sampler)
        self.assertTrue((self.root/'atlas_error.png').exists())
        tree=ast.parse((ROOT/'xgfit/xgfit/base.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        run=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        env={}
        module=ast.parse('from __future__ import annotations');module.body.append(run)
        exec(compile(module,'<run-wrapper>','exec'),env)
        original=ValueError('original fit error')
        def fail(*args):
            f.checkpoint_plot_state(f.gmodel)
            raise original
        f._run=fail
        f.make_error_atlas=Mock(side_effect=RuntimeError('plot failed'))
        with self.assertRaises(ValueError) as context:
            env['run'](f)
        self.assertIs(context.exception,original)
        f.make_error_atlas.assert_called_once()
        self.assertIsNone(f._plot_checkpoint)

    def test_one_g_terminal_plot_and_checkpoint_without_samples(self):
        f=self.fitter
        f.checkpoint_plot_state(f.gmodel_1G,include_samples=False)
        self.assertIsNone(f._plot_checkpoint['sampler'])
        f.make_atlas(gmodel=f.gmodel_1G,intermediate=False)
        self.assertEqual((self.root/'atlas.png').read_text(),'1')

    def test_display_subsampling_keeps_chain_and_original_step_axis(self):
        path=ROOT/'xgfit/plotter/plotter.py'
        fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)
                and n.name=='select_walk_plot_samples')
        env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
        raw=np.arange(1201*40*5,dtype=float).reshape(1201,40,5)
        before=raw.copy()
        display,steps=env[fn.name](raw,1000,30)
        self.assertEqual(display.shape,(1000,30,5))
        self.assertEqual(steps[0],0);self.assertEqual(steps[-1],1200)
        np.testing.assert_array_equal(raw,before)
        np.testing.assert_array_equal(display[:,0],raw[steps,0])
        np.testing.assert_array_equal(display[:,-1],raw[steps,-1])

    def test_config_defaults_and_validation(self):
        self.assertFalse(FitConfig().intermediate_plots)
        for changes in ({'walk_plot_max_steps':1},{'walk_plot_max_walkers':0}):
            with self.assertRaisesRegex(ValueError,'walk_plot'):
                replace(PipelineConfig(),fit=replace(FitConfig(),**changes)).validate()


if __name__=='__main__':unittest.main()

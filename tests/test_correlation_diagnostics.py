"""Disabled diagnostics perform no robust fitting and retain the CSV schema."""
import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.config import FitConfig
from pipeline.yaml_config import load_config


def load_helper(**extra):
    path=ROOT/'xgfit/xgfit/resampling.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)
              and n.name=='fill_correlation_diagnostics')
    env=dict(np=np,**extra)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env[node.name]


class CorrelationDiagnosticsTest(unittest.TestCase):
    def test_default_skips_calculation_and_clears_old_diagnostics(self):
        trim=Mock(side_effect=AssertionError('trimming must be skipped'))
        corr=Mock(side_effect=AssertionError('correlation must be skipped'))
        calculate=load_helper(trim_outlier_mahalanobis=trim,spearmanr=corr)
        frame=pd.DataFrame([dict(F21=1.2,S21=12.,Reliable='Y',rs_F21S21=.9)])
        samples=np.array([[1.,12.],[2.,13.],[3.,14.]])
        before=samples.copy()
        calculate(frame,['F21','S21'],samples)
        trim.assert_not_called();corr.assert_not_called()
        self.assertTrue(np.isnan(frame.loc[0,'rs_F21S21']))
        self.assertTrue(np.isnan(frame.loc[0,'p_rs_F21S21']))
        self.assertEqual(frame.loc[0,'F21'],1.2)
        self.assertEqual(frame.loc[0,'Reliable'],'Y')
        np.testing.assert_array_equal(samples,before)

    def test_enabled_preserves_trimmed_spearman_calculation(self):
        trim=Mock(side_effect=lambda x,y:(x[:-1],y[:-1]))
        corr=Mock(wraps=spearmanr)
        calculate=load_helper(trim_outlier_mahalanobis=trim,spearmanr=corr)
        samples=np.array([[1.,4.,2.],[2.,3.,3.],[3.,2.,1.],[4.,1.,4.],[99.,99.,99.]])
        before=samples.copy();frame=pd.DataFrame([{}]);names=['F21','F22','S21']
        calculate(frame,names,samples,enabled=True)
        self.assertEqual(trim.call_count,3);self.assertEqual(corr.call_count,3)
        for i,j in [(0,1),(0,2),(1,2)]:
            r,p=spearmanr(samples[:-1,i],samples[:-1,j])
            self.assertAlmostEqual(frame.loc[0,f'rs_{names[i]}{names[j]}'],r)
            self.assertAlmostEqual(frame.loc[0,f'p_rs_{names[i]}{names[j]}'],p)
        np.testing.assert_array_equal(samples,before)

    def test_default_and_all_yaml_examples_are_disabled(self):
        self.assertFalse(FitConfig().correlation_diagnostics)
        for path in (ROOT/'configs').glob('*.yaml'):
            with self.subTest(path=path.name):
                self.assertFalse(load_config(path).fit.correlation_diagnostics)


if __name__=='__main__':unittest.main()

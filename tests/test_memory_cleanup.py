"""Bounded residual draws and resource cleanup preserve outputs and RNG order."""
import ast
import gc
import logging
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.worker import close_stack_logger


def helper(name):
    path=ROOT/'xgfit/xgfit/resampling.py'
    fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
    return env[name]


class MemoryCleanupTest(unittest.TestCase):
    def test_noise_batches_preserve_draws_and_global_rng(self):
        noise_rows=helper('residual_noise_rows')
        state=np.random.get_state()
        try:
            residuals=np.linspace(-1,1,19)
            np.random.seed(656)
            expected=np.random.choice(residuals,size=(301,23),replace=True)
            expected_after=np.random.choice(10000,size=20)
            np.random.seed(656)
            with patch.object(np.random, 'choice', wraps=np.random.choice) as draw:
                rows=noise_rows(residuals,301,23,batch_size=32)
            self.assertTrue(all(call.kwargs['size'][0] <= 32 for call in draw.call_args_list))
            actual=[];indices=[]
            for noise in rows:
                actual.append(noise.copy())
                # Exercise optimizer/start-index draws interleaved with replay.
                if len(indices)<20:indices.append(np.random.choice(10000))
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(indices,expected_after)
        finally:np.random.set_state(state)

    def test_chain_indexing_matches_flatten_without_allocating_flat_copy(self):
        chain=np.arange(13*7*5).reshape(13,7,5)[::2]
        expected=chain.reshape(-1,5)
        for index in (0,1,12,len(expected)-1):
            np.testing.assert_array_equal(chain[index//7,index%7],expected[index])

    def test_logger_handlers_and_registry_are_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger=logging.getLogger('sns.memory_cleanup_test')
            handler=logging.FileHandler(Path(tmp)/'test.log');logger.addHandler(handler)
            stack=SimpleNamespace(logger=logger)
            close_stack_logger(stack)
            self.assertIsNone(handler.stream)
            self.assertIsNone(stack.logger)
            self.assertNotIn(logger.name,logging.Logger.manager.loggerDict)
            close_stack_logger(stack)

    def test_cleanup_releases_checkpoint_and_preserves_result_tables(self):
        path=ROOT/'xgfit/xgfit/cleanup.py'
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef))
        env={'gc':gc,'plt':SimpleNamespace(close=lambda *a:None)}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),env)
        obj=env['CleanupMixin']();array=np.zeros(10000);ref=weakref.ref(array)
        obj._plot_checkpoint={'sampler':array};obj.list_NHI_=array;obj.sampler=array
        obj.df={'answer':42};obj.df_params={'Reliable':'Y'}
        del array
        obj.cleanup(aggressive=False)
        self.assertIsNone(ref())
        self.assertEqual(obj.df,{'answer':42})
        self.assertEqual(obj.df_params,{'Reliable':'Y'})


if __name__=='__main__':unittest.main()

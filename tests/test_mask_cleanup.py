"""Only successful opted-in runs remove masks they used or generated."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PipelineConfig
from pipeline.runner import run_pipeline
from pipeline.yaml_config import load_config


class MaskCleanupTest(unittest.TestCase):
    def test_success_failure_disabled_and_dry_run(self):
        cases = [(True,False,False),(False,False,False),(True,True,False),(True,False,True)]
        for enabled, fail, dry in cases:
            with self.subTest(enabled=enabled,fail=fail,dry=dry), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                galaxy=root/'input'/'TEST'
                galaxy.mkdir(parents=True)
                (galaxy/'cube.fits').touch()
                (galaxy/'hermite.npy').touch()
                maskdir=galaxy/'NGFIT/mask'
                maskdir.mkdir(parents=True)
                unrelated=maskdir/'mask_unrelated.fits'
                unrelated.write_text('keep')
                companion=maskdir/'mask_O1.0Ropt.fits'
                companion.write_text('old companion')
                selected=maskdir/'mask_IO1.0Ropt.fits'
                config=PipelineConfig(data_dir=galaxy.parent,output_dir=root/'output',workers=1,dry_run=dry)
                config=replace(config,masks=replace(config.masks,regions=('inner',),cleanup_after_run=enabled))
                def generate(job,*args,**kwargs):
                    job.mask.write_text('selected')
                    companion.write_text('rewritten companion mask')
                def fit(task):
                    self.assertTrue(selected.exists())
                    if fail:
                        raise RuntimeError('fit failed')
                    frame=pd.DataFrame({'Name':['TEST'],'suffix':[task.stack.suffix]})
                    for kind in ('gfit','para'):
                        frame.to_string(task.temp/f'TEST{task.stack.suffix}_plotfitter_temp_{kind}.csv',index=False)
                    return task.stack.key
                with patch('pipeline.runner.validate_catalogs',side_effect=lambda c,g:g), \
                     patch('pipeline.runner.write_geometry'), \
                     patch('pipeline.runner.prepare_mask',side_effect=generate), \
                     patch('pipeline.worker.fit_stack',side_effect=fit):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'fit failed'):
                            run_pipeline(config)
                    else:
                        run_pipeline(config)
                self.assertEqual(unrelated.read_text(),'keep')
                if dry:
                    self.assertFalse(selected.exists())
                    self.assertEqual(companion.read_text(),'old companion')
                elif enabled and not fail:
                    self.assertFalse(selected.exists())
                    self.assertFalse(companion.exists())
                else:
                    self.assertTrue(selected.exists())
                    self.assertTrue(companion.exists())

    def test_all_yaml_examples_expose_boolean_option(self):
        configs=Path(__file__).resolve().parents[1]/'configs'
        for path in configs.glob('*.yaml'):
            with self.subTest(path=path.name):
                self.assertTrue(load_config(path).masks.cleanup_after_run)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.yaml'
            path.write_text('masks:\n  cleanup_after_run: false\n')
            self.assertFalse(load_config(path).masks.cleanup_after_run)
            path.write_text('masks:\n  cleanup_after_run: "false"\n')
            with self.assertRaises(ValueError):
                load_config(path)


if __name__=='__main__':
    unittest.main()

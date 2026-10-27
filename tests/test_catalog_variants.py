"""Numbered cube variants share catalog geometry, preserving job identities."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.catalogs import resolve_catalog_name
from pipeline.config import PipelineConfig
from pipeline.models import GalaxyInput
from pipeline.optical import read_optical_geometry
from pipeline.planning import validate_catalogs, plan_jobs
from pipeline.masks import prepare_mask, write_geometry


class CatalogVariantsTest(unittest.TestCase):
    def test_exact_name_precedes_numeric_fallback(self):
        self.assertEqual(resolve_catalog_name('VCC656_1', {'VCC656'}), 'VCC656')
        self.assertEqual(resolve_catalog_name('VCC656_1', {'VCC656','VCC656_1'}), 'VCC656_1')
        self.assertEqual(resolve_catalog_name('VCC656_other', {'VCC656'}), 'VCC656_other')
        self.assertEqual(resolve_catalog_name('VCC656_1', {'OTHER'}), 'VCC656_1')

    def test_variant_validation_masks_and_output_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root/'catalog.txt'
            catalog.write_text('objname Diam_ra Diam_dec Diam Diam_ba Diam_pa\nVCC656 180 10 20 0.5 0\n')
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [16,16]
            wcs.wcs.crval = [180,10]
            wcs.wcs.cdelt = [-1/3600,1/3600]
            wcs.wcs.ctype = ['RA---TAN','DEC--TAN']
            galaxies=[]
            for name in ('VCC656_1','VCC656_2'):
                folder=root/name
                folder.mkdir()
                cube=folder/'cube.fits'
                fits.writeto(cube,np.zeros((31,31)),wcs.to_header())
                galaxies.append(GalaxyInput(name,cube,None,None,'HERMOM'))
            config=PipelineConfig()
            masks=replace(config.masks,catalog=catalog,regions=('whole','inner','angle'),angles=(90,),angle_widths=(120,))
            config=replace(config,masks=masks)
            self.assertEqual(validate_catalogs(config,galaxies),galaxies)
            jobs=plan_jobs(config,galaxies)
            for job in jobs:
                prepare_mask(job,masks)
                if job.mask is not None:
                    self.assertTrue(job.mask.is_file())
                    self.assertEqual(job.mask.parents[2].name,job.galaxy.name)
            self.assertEqual({j.key[0] for j in jobs},{'VCC656_1','VCC656_2'})
            write_geometry(masks,galaxies,root)
            import pandas as pd
            geometry=pd.read_csv(root/'optical_geometry.csv')
            self.assertEqual(set(geometry.Name),{'VCC656_1','VCC656_2'})
            catalog.write_text(catalog.read_text()+'VCC656_1 181 11 22 0.6 10\n')
            self.assertEqual(read_optical_geometry(catalog,'VCC656_1')['Diam_ra'],181)

    def test_duplicate_base_entry_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'catalog.txt'
            path.write_text('objname Diam_ra Diam_dec\nVCC656 180 10\nVCC656 180 10\n')
            with self.assertRaisesRegex(ValueError,'found 2'):
                read_optical_geometry(path,'VCC656_1',center_only=True)

    def test_one_valid_row_is_selected_and_ambiguous_rows_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'catalog.txt'
            header = 'objname Diam_ra Diam_dec Diam Diam_ba Diam_pa\n'
            good = 'VCC465 185.324097 11.511072 202.399994 0.38 165\n'
            invalid = 'VCC465 NaN NaN NaN NaN NaN\n'
            galaxy = GalaxyInput('VCC465', root/'cube.fits', None, None, 'HERMOM')
            base = PipelineConfig()
            config = replace(base, masks=replace(base.masks, catalog=path, regions=('inner',)))
            for rows in (good+invalid, invalid+good):
                path.write_text(header+rows)
                self.assertEqual(read_optical_geometry(path,'VCC465')['Diam'],202.399994)
                self.assertEqual(validate_catalogs(config,[galaxy]),[galaxy])
                self.assertEqual(read_optical_geometry(path,'VCC465_1')['Diam_pa'],165)
            path.write_text(header+invalid+invalid)
            self.assertEqual(validate_catalogs(config,[galaxy]),[])
            path.write_text(header+good+good)
            with self.assertRaisesRegex(ValueError,'found 2'):
                validate_catalogs(config,[galaxy])

    def test_validity_depends_on_required_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'catalog.txt'
            path.write_text('objname Diam_ra Diam_dec Diam Diam_ba Diam_pa\n'
                            'TEST 180 10 20 0.5 0\nTEST 181 11 NaN NaN NaN\n')
            self.assertEqual(read_optical_geometry(path,'TEST')['Diam_ra'],180)
            with self.assertRaisesRegex(ValueError,'found 2'):
                read_optical_geometry(path,'TEST',center_only=True)


if __name__=='__main__':
    unittest.main()

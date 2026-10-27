"""Common-kernel numerical equivalence, cache reuse, and data lifetime checks."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch

import numpy as np
from test_broad_width_mapping import Gmodel, map_params, numerics
from importlib import import_module
kernels = import_module(numerics.__name__ + '.kernels')


class SharedKernelsTest(unittest.TestCase):
    def test_one_g_variants_match_physical_likelihood(self):
        rng=np.random.default_rng(71)
        for fixed in (False,True):
            names=['F1','V1','S1'] + ([] if fixed else ['B1'])
            x=np.linspace(-50,50,51);y=rng.normal(0,.01,len(x));errors=np.full(len(x),.01)
            model=Gmodel(x,y,errors,names,{'F1':[0,3],'V1':[-5,5],'S1':[1,30],'B1':[-.05,.05]})
            model.BB=.003
            lp,obj=(model.make_log_prob_1G_B1x() if fixed else model.make_log_prob_1G())
            for u in rng.uniform(.05,.95,(20,len(names))):
                p=dict(zip(names,map_params(u,model,'u->x')))
                predicted=p['F1']/(p['S1']*np.sqrt(2*np.pi))*np.exp(-.5*((x-p['V1'])/p['S1'])**2)+p.get('B1',model.BB)
                expected=-.5*np.sum(((y-predicted)/errors)**2)
                self.assertAlmostEqual(lp(u),expected,places=9)
                self.assertAlmostEqual(obj(u,y),-expected,places=9)
            invalid=np.full(len(names),.5);invalid[0]=-1.
            self.assertEqual(lp(invalid),-np.inf)

    def test_different_data_and_array_layouts_reuse_one_signature(self):
        count=None
        for n in (13,21,35):
            raw=np.linspace(-30,30,n*2,dtype=np.float32)[::2]
            raw.setflags(write=False)
            m=Gmodel(raw,np.zeros(n,dtype=np.float32),np.full(n,.01),['F1','V1','S1'],
                     {'F1':[0,3],'V1':[-5,5],'S1':[1,30]})
            m.BB=.01
            lp,_=m.make_log_prob_1G_B1x()
            self.assertTrue(np.isfinite(lp(np.array([.4,.5,.3],dtype=np.float32))))
            self.assertEqual(m.x.dtype,np.float64)
            self.assertTrue(m.x.flags.c_contiguous and m.x.flags.writeable)
            if count is None:count=len(kernels.log_probability.signatures)
            self.assertEqual(len(kernels.log_probability.signatures),count)
        with self.assertRaises(ValueError):lp([.5,.5])

    def test_kernel_retains_no_task_arrays(self):
        refs=[]
        for n in range(17,37):
            m=Gmodel(np.linspace(-10,10,n),np.zeros(n),np.ones(n),['F1','V1','S1'],
                     {'F1':[0,3],'V1':[-5,5],'S1':[1,30]})
            m.BB=0.
            lp,obj=m.make_log_prob_1G_B1x()
            lp([.5,.5,.5])
            refs.extend(weakref.ref(a) for a in (m.x,m.y,m.e_y,m.low,m.hih,m.spn))
            del lp,obj,m
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_disk_cache_loaded_in_new_process_with_new_data(self):
        script='''
import sys,json,numpy as np
from test_broad_width_mapping import Gmodel,numerics
from importlib import import_module
k=import_module(numerics.__name__+'.kernels')
n=int(sys.argv[1]);x=np.linspace(-10,10,n)
m=Gmodel(x,np.full(n,n/1000.),np.ones(n),['F1','V1','S1'],{'F1':[0,3],'V1':[-5,5],'S1':[1,30]});m.BB=0.
lp,_=m.make_log_prob_1G_B1x()
print(json.dumps({'value':lp([.5,.5,.5]),'signatures':len(k.log_probability.signatures)}))
'''
        with tempfile.TemporaryDirectory() as tmp:
            env=os.environ.copy();env['NUMBA_CACHE_DIR']=tmp;env['NUMBA_DEBUG_CACHE']='1'
            tests=str(Path(__file__).resolve().parent)
            env['PYTHONPATH']=tests+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
            first=subprocess.run([sys.executable,'-c',script,'17'],env=env,capture_output=True,text=True,check=True,timeout=90)
            second=subprocess.run([sys.executable,'-c',script,'31'],env=env,capture_output=True,text=True,check=True,timeout=90)
            self.assertIn('[cache] data saved',first.stdout)
            self.assertIn('[cache] data loaded',second.stdout)
            a=json.loads(first.stdout.splitlines()[-1]);b=json.loads(second.stdout.splitlines()[-1])
            self.assertEqual(a['signatures'],1);self.assertEqual(b['signatures'],1)
            self.assertNotEqual(a['value'],b['value'])

    def test_warmup_uses_short_lived_process(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from pipeline.kernel_warmup import prepare_kernel_cache
        with patch('pipeline.kernel_warmup.subprocess.run') as run:
            prepare_kernel_cache()
        args,kwargs=run.call_args
        self.assertEqual(args[0],[sys.executable,'-m','pipeline.kernel_warmup'])
        self.assertTrue(kwargs['check'])
        self.assertIn('PYTHONPATH',kwargs['env'])


if __name__=='__main__':unittest.main()

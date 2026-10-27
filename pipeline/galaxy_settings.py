"""Galaxy-specific spectral and velocity-field settings."""
import astropy.units as u

dict_config_stack = {
    'NGC1569':{
        'specmask':{
            'from':-20*(u.km/u.s),
            'to'  :20*(u.km/u.s),
            'mask':'inner',
            'mode':'velocity'
        },
        'correct_secondary_vf':False
    },
    'AGC224248':{
        'specmask':{
            'from':0,
            'to'  :32,
            'mask':'inner',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'AGC225847':{
        'specmask':{
            'from':60,
            'to'  :120,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'AGC226178':{
        'specmask':{
            'from':25,
            'to'  :120,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC130':{
        'specmask':{
            'from':0,
            'to'  :30,
            'mask':'inner',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC152':{
        # 'specmask':{
        #     'from':40,
        #     'to'  :120,
        #     'mask':'outer',
        #     'mode':'channel'
        # },
        'specmask':None,
        'correct_secondary_vf':False
    },
    'VCC169':{
        'specmask':None,
        'correct_secondary_vf':False
    },
    'VCC309':{
        'specmask':{
            'from':0,
            'to'  :140,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC328':{
        'specmask':{
            'from':20,
            'to'  :140,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC329':{
        'specmask':{
            'from':30,
            'to'  :200,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC331':{
        'specmask':{
            'from':20,
            'to'  :120,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC340':{
        'specmask':{
            'from':0,
            'to'  :50,
            'mask':'inner',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC379':{
        'specmask':{
            'from':50,
            'to'  :230,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC381':{
        'specmask':{
            'from':20,
            'to'  :140,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC479':{
        'specmask':{
            'from':20,
            'to'  :110,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC566':{
        'specmask':{
            'from':10,
            'to'  :120,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC613':{
        'specmask':{
            'from':20,
            'to'  :185,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC656':{
        'specmask':None,
        'correct_secondary_vf':False
    },
    'VCC667':{
        'specmask':None,
        'correct_secondary_vf':False
    },
    'VCC693':{
        'specmask':{
            'from':20,
            'to'  :120,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC697':{
        'specmask':{
            'from':50,
            'to'  :400,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC699':{
        'specmask':{
            'from':20,
            'to'  :200,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC1091':{
        'specmask':{
            'from':80,
            'to'  :250,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC1098':{
        'specmask':{
            'from':60,
            'to'  :250,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC1142':{
        'specmask':{
            'from':30,
            'to'  :200,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC1411':{
        'specmask':{
            'from':0,
            'to'  :20,
            'mask':'inner',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC1744':{
        'specmask':{
            'from':0,
            'to'  :20,
            'mask':'inner',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC1778':{
        'specmask':None,
        'correct_secondary_vf':False
    },
    'VCC1992':{
        'specmask':{
            'from':0,
            'to'  :20,
            'mask':'inner',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC2006':{
        'specmask':{
            'from':220,
            'to'  :230,
            'mask':'inner',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC2034':{
        'specmask':{
            'from':0,
            'to'  :120,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCC2037':{
        'specmask':{
            'from':0,
            'to'  :20,
            'mask':'inner',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCCA055':{
        'specmask':{
            'from':30,
            'to'  :120,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },        
    'VCCA069':{
        'specmask':{
            'from':20,
            'to'  :120,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
    'VCCA070':{
        'specmask':{
            'from':20,
            'to'  :120,
            'mask':'outer',
            'mode':'channel'
        },
        'correct_secondary_vf':False
    },
}

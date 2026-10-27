Optical inner/outer stacking
============================

The pipeline is now organized in `pipeline/`. See [README.md](README.md) for
configuration, code layout, mask conventions, outputs, and validation.

The default is the VIVA_test optical experiment:

```sh
python3.13 pipe_superprofile.py --dry-run
python3.13 pipe_superprofile.py
```

To select the NEDLVS catalog explicitly:

```sh
python3.13 pipe_superprofile.py \
  --data-dir /home/mskim/workspace/research/data/VIVA/VIVA_test \
  --catalog /home/mskim/workspace/research/data/catalog/diameters_NEDLVS.csv \
  --workers 2
```

For `VCC307`, Ropt is 161.2683 arcsec. The original completed inner and outer
results remain in `research/data/VIVA/VIVA_test_2GFIT_HERMOM_Ropt/`.

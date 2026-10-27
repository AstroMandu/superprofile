"""Whole-galaxy initialization using the original pipeline's fit ratios."""
import numpy as np
import pandas as pd


def apply_whole_guess(fitter, task):
    if not task.settings.guess_from_whole or task.stack.region == 'whole':
        return
    galaxy = task.stack.galaxy.name
    path = task.output / 'info_stacked_GFIT.csv'
    if not path.exists():
        raise RuntimeError(f'Run whole first: {galaxy}')
    table = pd.read_csv(path, sep=r'\s+')
    rows = table.loc[(table['Name'] == galaxy) & (table['suffix'] == '_')]
    if len(rows) != 1:
        raise RuntimeError(f'Expected one whole-galaxy fit: {galaxy}')
    row = rows.iloc[0]
    if task.settings.model == '2GFIT':
        if not np.isfinite(row['SNR2']):
            raise RuntimeError(f'Whole-galaxy 2G fit is unavailable: {galaxy}')
        v21 = 0.0 if row['SNR21'] < 1 else row['V21']
        v22 = 0.0 if row['SNR22'] < 1 else row['V22']
        s22 = 3 * row['S1'] if row['SNR22'] < 1 else row['S22']
        fitter.dict_preguess = {
            'F21/F1': row['F21'] / row['F1'], 'F22/F1': row['F22'] / row['F1'],
            'A21/A1': row['A21'] / row['A1'], 'A22/A1': row['A22'] / row['A1'],
            'V21': v21, 'V22': v22, 'S21': row['S21'], 'S22': s22,
        }
    elif np.isfinite(row['SNR3']):
        fitter.dict_preguess = {
            **{f'F3{i}/F1': row[f'F3{i}'] / row['F1'] for i in (1, 2, 3)},
            **{f'S3{i}': row[f'S3{i}'] for i in (1, 2, 3)},
        }
    elif np.isfinite(row['SNR2']):
        fitter.dict_preguess = {
            'F31/F1': row['F21'] / row['F1'], 'F32/F1': row['F22'] / row['F1'],
            'F33/F1': 0.0001 * row['F1'],
            'V21': 0.0 if row['SNR21'] < 1 else row['V21'],
            'V22': 0.0 if row['SNR22'] < 1 else row['V22'],
            'S31': row['S21'], 'S32': row['S22'], 'S33': row['S22'] + 0.1,
        }
    else:
        raise RuntimeError(f'Whole-galaxy multi-Gaussian fit is unavailable: {galaxy}')

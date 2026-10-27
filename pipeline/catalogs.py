"""Identify supported catalog layouts from their header columns."""
from pathlib import Path


OPTICAL_COLUMNS = {'objname', 'Diam_ra', 'Diam_dec'}


def resolve_catalog_name(name, catalog_names):
    """Prefer exact entries; numbered data variants may share base geometry."""
    if name in catalog_names:
        return name
    base, separator, variant = name.rpartition('_')
    if separator and variant.isdigit() and base in catalog_names:
        return base
    return name


def catalog_kind(path):
    """Return ``optical`` for NEDLVS, otherwise ``legacy`` for Name tables."""
    path = Path(path)
    with path.open(encoding='utf-8-sig') as stream:
        header = next((line.split() for line in stream
                       if line.strip() and not line.lstrip().startswith('#')), [])
    columns = set(header)
    if OPTICAL_COLUMNS <= columns:
        return 'optical'
    if 'Name' in columns:
        return 'legacy'
    raise ValueError(
        f'{path}: unrecognized catalog header; expected NEDLVS columns '
        f'{", ".join(sorted(OPTICAL_COLUMNS))} or a legacy Name column'
    )

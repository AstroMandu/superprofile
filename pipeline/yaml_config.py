"""Safe, typed YAML loading with paths relative to the configuration file."""
from dataclasses import fields, replace, is_dataclass
from pathlib import Path
from types import UnionType
from typing import get_args, get_origin, get_type_hints


def _value(value, annotation, base, name):
    origin = get_origin(annotation)
    if origin is UnionType:
        if value is None and type(None) in get_args(annotation):
            return None
        annotation = next(item for item in get_args(annotation) if item is not type(None))
        origin = get_origin(annotation)
    if is_dataclass(annotation):
        return _update(annotation(), value, base, name)
    if annotation is Path:
        if not isinstance(value, str):
            raise ValueError(f'{name}: expected a path string')
        path = Path(value).expanduser()
        return (base / path).resolve() if not path.is_absolute() else path.resolve()
    if origin is tuple:
        if name == 'masks.angles' and isinstance(value, dict):
            if set(value) != {'start', 'stop', 'step'}:
                raise ValueError(f'{name}: range requires exactly start, stop, step')
            if any(type(item) is not int for item in value.values()):
                raise ValueError(f'{name}: range start, stop, step must be integers')
            if value['step'] == 0:
                raise ValueError(f'{name}: range step must not be zero')
            angles = range(value['start'], value['stop'], value['step'])
            if not angles or angles[0] < 0 or angles[0] >= 360 or angles[-1] < 0 or angles[-1] >= 360:
                raise ValueError(f'{name}: range must produce angles in [0, 360) and must not be empty')
            value = list(angles)
        if not isinstance(value, list):
            raise ValueError(f'{name}: expected a YAML list')
        item_type = get_args(annotation)[0]
        return tuple(_value(item, item_type, base, f'{name}[{index}]') for index, item in enumerate(value))
    if annotation is float and type(value) in (int, float):
        return float(value)
    if type(value) is not annotation:
        raise ValueError(f'{name}: expected {annotation.__name__}, got {type(value).__name__}')
    return value


def _update(config, values, base, section):
    if not isinstance(values, dict):
        raise ValueError(f'{section}: expected a YAML mapping')
    hints = get_type_hints(type(config))
    unknown = set(values) - {field.name for field in fields(config)}
    if unknown:
        raise ValueError(f'{section}: unknown settings: {", ".join(sorted(map(str, unknown)))}')
    updates = {}
    for name, value in values.items():
        current = getattr(config, name)
        if is_dataclass(current):
            updates[name] = _update(current, value, base, name)
        else:
            updates[name] = _value(value, hints[name], base, f'{section}.{name}')
    # Changing samples should not inherit a preset's named galaxy selection.
    if section == 'config' and 'data_dir' in values and 'galaxies' not in values:
        updates['galaxies'] = None
    return replace(config, **updates)


def load_config(path, preset_override=None):
    import yaml
    from .config import preset
    path = Path(path).expanduser().resolve()
    try:
        values = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ValueError(f'{path}: invalid YAML: {error}') from error
    if not isinstance(values, dict):
        raise ValueError(f'{path}: expected a YAML mapping')
    values = dict(values)
    name = values.pop('preset', 'viva-test')
    if not isinstance(name, str):
        raise ValueError('preset: expected a string')
    # CLI overrides are applied after loading, before final validation.
    return _update(preset(preset_override or name), values, path.parent, 'config')

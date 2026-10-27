"""Picklable data passed between planning, execution, and fit workers."""
from dataclasses import dataclass
from pathlib import Path

from .config import FitConfig


@dataclass(frozen=True)
class GalaxyInput:
    name: str
    cube: Path
    classified: Path | None
    secondary_vf: Path | None
    secondary_mode: str


@dataclass(frozen=True)
class StackJob:
    galaxy: GalaxyInput
    region: str
    suffix: str
    mask: Path | None = None
    multiplier: float | None = None
    angle: int | None = None
    angle_width: int | None = None
    radial_region: str | None = None
    inner_arcsec: float | None = None
    outer_arcsec: float | None = None
    beam_arcsec: float | None = None

    @property
    def key(self):
        return self.galaxy.name, self.suffix

    @property
    def mask_value(self):
        """Shared IO files select inner=1 or outer=0; other masks use finite pixels."""
        radial = self.radial_region if self.region == 'angle' else self.region
        return {'inner': 1, 'outer': 0}.get(radial)


@dataclass(frozen=True)
class FitJob:
    stack: StackJob
    settings: FitConfig
    output: Path
    temp: Path
    show_progress: bool = False


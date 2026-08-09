"""Load configs/base.yaml once. Import as: from prometheus.config import cfg"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def project_root() -> Path:
    """Repository root (parent of src/)."""
    return Path(__file__).resolve().parents[2]


class ProjectCfg(BaseModel):
    name: str
    version: str
    description: str = ""


class PathsCfg(BaseModel):
    root: str = "."
    data: str = "data"
    raw: str = "data/raw"
    firms_raw: str = "data/raw/firms"
    firms_archives: str = "data/raw/firms/archives"
    firms_chunks: str = "data/raw/firms/chunks"
    gee_raw: str = "data/raw/gee"
    static: str = "data/static"
    cube: str = "data/cube"
    models: str = "data/models"
    runs: str = "runs"
    docs: str = "docs"
    configs: str = "configs"
    nepal_mask: str = "data/static/nepal_mask_1km_roiAligned.tif"
    elevation: str = "data/static/elevation_static_srtm.tif"
    slope: str = "data/static/slope_static_srtm.tif"

    def resolve(self, key: str) -> Path:
        return (project_root() / getattr(self, key)).resolve()


class RoiCfg(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @property
    def bbox(self) -> list[float]:
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]


class GridCfg(BaseModel):
    height: int
    width: int
    crs: str
    transform: list[float]
    nodata: float = -9999.0
    mask_valid_value: int = 1

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


class YearsCfg(BaseModel):
    train_start: int
    train_end: int
    all: list[int]
    climatology: list[int]


class SeasonCfg(BaseModel):
    months: list[int]
    start_month: int = 1
    start_day: int = 1
    end_month: int = 5
    end_day: int = 31


class LabelsCfg(BaseModel):
    satellites: list[str]
    api_endpoint: str = "area"
    modis_confidence_min: int = 50
    viirs_confidence: list[str]
    drop_type_nonzero: bool = True
    dilate_pixels: int = 1
    bbox: list[float]


class FeaturesCfg(BaseModel):
    weather_daily: list[str] = Field(default_factory=list)
    vegetation: list[str] = Field(default_factory=list)
    thermal: list[str] = Field(default_factory=list)
    terrain_static: list[str] = Field(default_factory=list)
    landcover_static: list[str] = Field(default_factory=list)
    human_static: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    temporal: list[str] = Field(default_factory=list)
    forest_worldcover_classes: list[int] = Field(default_factory=list)

    @property
    def all_names(self) -> list[str]:
        return (
            self.weather_daily
            + self.vegetation
            + self.thermal
            + self.terrain_static
            + self.landcover_static
            + self.human_static
            + self.history
            + self.temporal
        )


class ModelingCfg(BaseModel):
    positive_negative_ratio: int = 20
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.8
    early_stopping_rounds: int = 50
    random_seed: int = 42
    horizons: list[int] = Field(default_factory=lambda: [1, 7])


class CvCfg(BaseModel):
    scheme: str = "leave_one_year_out"
    years: list[int]
    primary_metric: str = "pr_auc"
    top_k_fractions: list[float] = Field(default_factory=lambda: [0.05, 0.10])


class RiskClassesCfg(BaseModel):
    names: list[str]
    quantiles: list[float]


class Settings(BaseModel):
    project: ProjectCfg
    paths: PathsCfg
    roi: RoiCfg
    grid: GridCfg
    years: YearsCfg
    season: SeasonCfg
    labels: LabelsCfg
    features: FeaturesCfg
    modeling: ModelingCfg
    cv: CvCfg
    risk_classes: RiskClassesCfg
    regions: list[str] = Field(default_factory=list)

    @property
    def years_list(self) -> list[int]:
        return list(self.years.all)

    @property
    def season_months(self) -> list[int]:
        return list(self.season.months)

    @property
    def root(self) -> Path:
        return project_root()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


@lru_cache(maxsize=1)
def load_settings(config_name: str = "base.yaml") -> Settings:
    path = project_root() / "configs" / config_name
    if not path.is_file():
        raise FileNotFoundError(f"Missing config: {path}")
    return Settings.model_validate(_load_yaml(path))


class _CfgProxy:
    def __getattr__(self, name: str) -> Any:
        settings = load_settings()
        if name == "years":
            return settings.years_list
        if name == "season_months":
            return settings.season_months
        if hasattr(settings, name):
            return getattr(settings, name)
        raise AttributeError(f"Settings has no attribute {name!r}")

    def settings(self) -> Settings:
        return load_settings()

    def reload(self) -> Settings:
        load_settings.cache_clear()
        return load_settings()


cfg = _CfgProxy()

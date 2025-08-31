from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Vars:
    aod: str = "aod"
    lat: str = "lat"
    lon: str = "lon"
    ra: str = "RA"
    za: str = "ZA"
    vz: str = "VZ"
    tr: str = "tr"
    rad: str = "rad"
    uv: str = "uv"
    vis: str = "vis"


@dataclass
class TrainConfig:
    nc_path: str
    save_dir: str = "./outputs"
    sz: int = 32
    testratio: float = 0.2
    modes: list[str] = None
    gap_sizes: list[int] = None
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    var: Vars = Vars()


@staticmethod
def from_args_and_yaml(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--nc-path", type=str)
    ap.add_argument("--save-dir", type=str, default="./outputs")
    ap.add_argument("--sz", type=int, default=32)
    ap.add_argument("--testratio", type=float, default=0.2)
    ap.add_argument("--modes", nargs="+", default=["univariate","multivariate"])
    ap.add_argument("--gap-sizes", nargs="+", type=int, default=[2,5,15])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    # vars
    ap.add_argument("--var-aod", default="aod")
    ap.add_argument("--var-lat", default="lat")
    ap.add_argument("--var-lon", default="lon")
    ap.add_argument("--var-ra", default="RA")
    ap.add_argument("--var-za", default="ZA")
    ap.add_argument("--var-vz", default="VZ")
    ap.add_argument("--var-tr", default="tr")
    ap.add_argument("--var-rad", default="rad")
    ap.add_argument("--var-uv", default="uv")
    ap.add_argument("--var-vis", default="vis")
    args = ap.parse_args(argv)


    cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}


    # merge precedence: CLI > YAML > defaults
    def pick(key, default=None):
        return getattr(args, key) if getattr(args, key) is not None else cfg.get(key, default)


    var_cfg = cfg.get("var", {})
    vars = Vars(
        aod=pick("var_aod") or var_cfg.get("aod", "aod"),
        lat=pick("var_lat") or var_cfg.get("lat", "lat"),
        lon=pick("var_lon") or var_cfg.get("lon", "lon"),
        ra=pick("var_ra") or var_cfg.get("ra", "RA"),
        za=pick("var_za") or var_cfg.get("za", "ZA"),
        vz=pick("var_vz") or var_cfg.get("vz", "VZ"),
        tr=pick("var_tr") or var_cfg.get("tr", "tr"),
        rad=pick("var_rad") or var_cfg.get("rad", "rad"),
        uv=pick("var_uv") or var_cfg.get("uv", "uv"),
    )

    return TrainConfig(
        nc_path=pick("nc_path"),
        save_dir=pick("save_dir", "./outputs"),
        sz=int(pick("sz", 32)),
        testratio=float(pick("testratio", 0.2)),
        modes=pick("modes", ["univariate","multivariate"]),
        gap_sizes=[int(x) for x in pick("gap_sizes", [2,5,15])],
        epochs=int(pick("epochs", 50)),
        batch_size=int(pick("batch_size", 32)),
        learning_rate=float(pick("learning_rate", 1e-3)),
        var=vars,
        )
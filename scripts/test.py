#!/usr/bin/env python3
import argparse
from aod_gapfill.config import TrainConfig
from aod_gapfill.eval import evaluate


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--nc-path", type=str, required=True, help="Path to test NetCDF (test_all.nc)")
    ap.add_argument("--load-dir", type=str, default="./outputs")
    ap.add_argument("--sz", type=int, default=32)
    ap.add_argument("--evaluate-gaps", nargs="+", type=int, default=[2,5,15])
    ap.add_argument("--scatter", action="store_true")
    ap.add_argument("--heatmap", action="store_true")
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
    args = ap.parse_args()


    # Build TrainConfig-like object for shared fields
    cfg = TrainConfig.from_args_and_yaml([
        "--nc-path", args.nc_path,
        "--save-dir", args.load_dir,
        "--sz", str(args.sz),
        "--var-aod", args.var_aod,
        "--var-lat", args.var_lat,
        "--var-lon", args.var_lon,
        "--var-ra", args.var_ra,
        "--var-za", args.var_za,
        "--var-vz", args.var_vz,
        "--var-tr", args.var_tr,
        "--var-rad", args.var_rad,
        "--var-uv", args.var_uv,
        "--var-vis", args.var_vis,
    ])


    evaluate(cfg, test_nc_path=args.nc_path, evaluate_gaps=args.evaluate_gaps,
            do_scatter=args.scatter, do_heatmap=args.heatmap)
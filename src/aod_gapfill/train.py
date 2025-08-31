from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau


from .config import TrainConfig
from .data import load_and_normalize, to_img4, split_indices, make_gapped
from .model import build_unet
from .metrics import RMSE




def train(cfg: TrainConfig):
    save_root = Path(cfg.save_dir)
    (save_root / "model").mkdir(parents=True, exist_ok=True)
    (save_root / "history").mkdir(parents=True, exist_ok=True)
    (save_root / "figs").mkdir(parents=True, exist_ok=True)


    varnames = [cfg.var.aod, cfg.var.lat, cfg.var.lon, cfg.var.ra, cfg.var.za,
    cfg.var.vz, cfg.var.tr, cfg.var.rad, cfg.var.uv, cfg.var.vis]
    dsnorm, minmax = load_and_normalize(cfg.nc_path, cfg.sz, varnames)


    Ntime = dsnorm.sizes["time"]
    trains, tests = split_indices(Ntime, cfg.testratio)


    y_all = to_img4(dsnorm, cfg.var.aod, cfg.sz)
    y_train = y_all[trains]
    y_val = y_all[tests]


    # multivariate stack if possible
    has_multi = all(v in dsnorm for v in varnames)
    x_multi_full = None
    if has_multi:
        arrs = [to_img4(dsnorm, v, cfg.sz) for v in varnames]
        x_multi_full = np.concatenate(arrs, axis=-1)


    for g in cfg.gap_sizes:
        # Univariate input
        x_uni = make_gapped(y_all, gap_size=g, miss_val=0.0)
        x_train_uni = x_uni[trains]
        x_val_uni = x_uni[tests]

        if "univariate" in cfg.modes:
            m = build_unet(input_ch=1, sz=cfg.sz)
            m.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate), loss=RMSE, metrics=[RMSE, "accuracy"])
            hist = m.fit(x_train_uni, y_train,
            validation_data=(x_val_uni, y_val),
            epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=1,
            callbacks=[ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-9)])
            m.save(save_root / "model" / f"model_one_var_noise{g}.keras")
            pd.DataFrame(hist.history).to_csv(save_root / "history" / f"training_historymodel_one_var_noise{g}.csv", index=False)


        if "multivariate" in cfg.modes and has_multi:
            x_multi = x_multi_full.copy()
            x_multi[...,0:1] = make_gapped(x_multi[...,0:1], gap_size=g, miss_val=0.0)
            m = build_unet(input_ch=x_multi.shape[-1], sz=cfg.sz)
            m.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate), loss=RMSE, metrics=[RMSE, "accuracy"])
            hist = m.fit(x_multi[trains], y_train,
            validation_data=(x_multi[tests], y_val),
            epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=1,
            callbacks=[ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-9)])
            m.save(save_root / "model" / f"model_all_var_noise{g}.keras")
            pd.DataFrame(hist.history).to_csv(save_root / "history" / f"training_historymodel_all_var_noise{g}.csv", index=False)
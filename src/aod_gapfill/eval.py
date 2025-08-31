from __future__ import annotations
from .metrics import RMSE, rmse_map, ssim_map, denorm
from .viz import save_tile




def evaluate(cfg: TrainConfig, test_nc_path: str, evaluate_gaps: list[int], do_scatter=False, do_heatmap=False):
    save_root = Path(cfg.save_dir)
    out_dir = save_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)


    varnames = [cfg.var.aod, cfg.var.lat, cfg.var.lon, cfg.var.ra, cfg.var.za,
    cfg.var.vz, cfg.var.tr, cfg.var.rad, cfg.var.uv, cfg.var.vis]
    dsnorm, minmax = load_and_normalize(test_nc_path, cfg.sz, varnames)


    y_true = to_img4(dsnorm, cfg.var.aod, cfg.sz)
    has_multi = all(v in dsnorm for v in varnames)
    x_multi_full = None
    if has_multi:
        arrs = [to_img4(dsnorm, v, cfg.sz) for v in varnames]
        x_multi_full = np.concatenate(arrs, axis=-1)


    results = []
    for gtest in evaluate_gaps:
        x_uni = make_gapped(y_true, gap_size=gtest, miss_val=0.0)
        for gtrain in [2,5,15]:
        mpath = save_root / "model" / f"model_one_var_noise{gtrain}.keras"
        if mpath.exists():
            m = load_model(mpath, custom_objects={"RMSE": RMSE})
            pred = m.predict(x_uni, verbose=0)
            vmin, vmax = minmax[cfg.var.aod]
            rmse = rmse_map(denorm(y_true[...,0], vmin, vmax), denorm(pred[...,0], vmin, vmax))
            ssimv = ssim_map(denorm(y_true[...,0], vmin, vmax), denorm(pred[...,0], vmin, vmax))
            results.append(pd.DataFrame({
                "mode": "univariate", "train_gap": gtrain, "test_gap": gtest,
                "rmse": rmse, "ssim": ssimv
            }))
            k = min(101, y_true.shape[0]-1)
            save_tile(x_uni[k,:,:,0], y_true[k,:,:,0], pred[k,:,:,0], f"Pred U (train={gtrain})", out_dir / f"tile_uni_train{gtrain}_test{gtest}.png")


    if has_multi:
        x_multi = x_multi_full.copy()
        x_multi[...,0:1] = make_gapped(x_multi[...,0:1], gap_size=gtest, miss_val=0.0)
        for gtrain in [2,5,15]:
            mpath = save_root / "model" / f"model_all_var_noise{gtrain}.keras"
            if mpath.exists():
                m = load_model(mpath, custom_objects={"RMSE": RMSE})
                pred = m.predict(x_multi, verbose=0)
                vmin, vmax = minmax[cfg.var.aod]
                rmse = rmse_map(denorm(y_true[...,0], vmin, vmax), denorm(pred[...,0], vmin, vmax))
                ssimv = ssim_map(denorm(y_true[...,0], vmin, vmax), denorm(pred[...,0], vmin, vmax))
                results.append(pd.DataFrame({
                "mode": "multivariate", "train_gap": gtrain, "test_gap": gtest,
                "rmse": rmse, "ssim": ssimv
                }))
                k = min(101, y_true.shape[0]-1)
                save_tile(x_multi[k,:,:,0], y_true[k,:,:,0], pred[k,:,:,0], f"Pred M (train={gtrain})", out_dir / f"tile_multi_train{gtrain}_test{gtest}.png")


    if results:
        df = pd.concat(results, ignore_index=True)
        df.to_csv(out_dir / "metrics_rmse_ssim.csv", index=False)
        agg = df.groupby(["mode","train_gap","test_gap"]).agg({"rmse":"mean","ssim":"mean"}).reset_index()
        agg.to_csv(out_dir / "metrics_agg.csv", index=False)
        print("Saved:", out_dir / "metrics_rmse_ssim.csv")
        print("Saved:", out_dir / "metrics_agg.csv")
    else:
        print("No models found in", save_root / "model")
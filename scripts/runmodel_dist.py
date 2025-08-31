#!/usr/bin/env python3
from aod_gapfill.config import TrainConfig
from aod_gapfill.train import train


if __name__ == "__main__":
    cfg = TrainConfig.from_args_and_yaml()
    train(cfg)
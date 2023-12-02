from dataclasses import dataclass


# from typing import Any


CATBOOST_RANDOM_SEED = 8
CATBOOST_SILENT_FLAG = True


@dataclass
class CatBoostModel:
    iterations: int
    max_depth: int
    learning_rate: float


@dataclass
class Params:
    model: CatBoostModel

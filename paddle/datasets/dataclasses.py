from dataclasses import dataclass
from typing import List, Optional
from pandas import DataFrame


@dataclass
class DataSplits:
    train: Optional[List[List[str]]]
    test: Optional[List[List[str]]]
    dev: Optional[List[List[str]]]


@dataclass
class DataSplitsOpinHuBank:
    train: Optional[DataFrame]
    test: Optional[DataFrame]
    dev: Optional[DataFrame]

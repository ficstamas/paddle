from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataSplits:
    train: Optional[List[List[str]]]
    test: Optional[List[List[str]]]
    dev: Optional[List[List[str]]]

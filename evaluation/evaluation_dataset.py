import abc

from typing import List, Tuple


class TableInfo:
    def __init__(self, path: str, weight: float = 1,
                 cut_range: Tuple[float, float] = (0.0, 1.1)):
        self.path = path
        self.weight = weight
        self.cut_range = cut_range

    @property
    def data_type(self):
        return self.path.split(':')[0]

    @property
    def data_dir_path(self):
        return self.path[len(self.data_type) + 1:]


class EvaluationDataset:
    def __init__(self, name: str, table: TableInfo):
        self.name = name
        self.table = table

    def is_synthetic(self):
        if self.table.data_type in ['example', 'kekos']:
            return False
        return True

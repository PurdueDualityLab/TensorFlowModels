import abc
from typing import Callable

class DatasetParser(abc.ABC):
    @abc.abstractmethod
    def unbatched_process_fn(self, is_training: bool) -> Callable[[dict], dict]:
        ...

    @abc.abstractmethod
    def batched_process_fn(self, is_training: bool) -> Callable[[dict], dict]:
        ...

    @abc.abstractmethod
    def build_gt(self, is_training: bool) -> Callable[[dict], dict]:
        ...

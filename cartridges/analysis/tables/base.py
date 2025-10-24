import os
from abc import ABC, abstractmethod
from pydrantic import RunConfig, ObjectConfig


class TableConfig(RunConfig, ObjectConfig):
    _pass_as_config: bool = True

    output_dir: str = os.path.join(os.getenv("HAYSTACKS_DIR"), "haystacks/communication/analysis/outputs")


    def run(self):
        table = self.instantiate()
        latex = table.build()

        print(f"---BEGIN TABLE--- \n\n{latex}\n\n---END TABLE---")

    


class Table(ABC):

    def __init__(self, config: TableConfig):
        self.config = config

    @abstractmethod
    def _prepare_data(self):
        raise NotImplementedError("Subclass must implement prepare_data method")

    @abstractmethod
    def build(self):
        raise NotImplementedError("Subclass must implement table method")
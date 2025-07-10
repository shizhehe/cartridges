import abc
from typing import List

from pydrantic import ObjectConfig


class Resource(abc.ABC):

    class Config(ObjectConfig):
        _pass_as_config = True
    
    async def setup(self):
        """This is an optional method that can be used to setup the resource.
        It is called before the first call to sample_prompt.
        """
        pass
    
    @abc.abstractmethod
    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        raise NotImplementedError()

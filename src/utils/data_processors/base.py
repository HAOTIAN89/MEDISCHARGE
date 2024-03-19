# This file defines the abstract class for utility classes.

from abc import ABC, abstractmethod
from typing import Callable

class BaseDataProcessor(ABC, Callable):
    """This class is the abstract base class for all data processors."""
    
    @abstractmethod   
    def __call__(self, data, **kwargs):
        """Processes the data."""
        pass
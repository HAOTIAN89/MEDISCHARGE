# This file contains the class used to load the data processors using their names.

from .classes import *

# Register the data processors here or using the <AutoDataProcessor.register> method.
AUTO_CLASSES = {
    "exploration": ExplorationDataProcessor
}

class AutoDataProcessor:
    """This class is used to load the data processors using their names like huggingface AutoClasses."""
    @staticmethod
    def load(name: str, *args, **kwargs) -> BaseDataProcessor:
        """Loads the a data processor using its name."""
        global AUTO_CLASSES
        try:
            processor = AUTO_CLASSES[name](*args, **kwargs)
        except:
            raise ValueError(f"Unregistered data processor name: {name}")
        return processor
    
    @staticmethod
    def register(name: str, cls):
        """Registers the class so that it can be loaded using the load method."""
        global AUTO_CLASSES
        AUTO_CLASSES[name] = cls
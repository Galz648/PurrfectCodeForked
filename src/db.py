
import abc
import pandas as pd
import uuid
from time import time

class DatabaseClientStrategy(abc.ABC):
    
    @abc.abstractmethod
    def save(self, data: pd.DataFrame) -> None:
        pass




class SaveJsonToFileStrategy(DatabaseClientStrategy):
    def save(self, data: pd.DataFrame) -> None:
        filename = f"posts_{time.now()}" + str(uuid.uuid4()) + ".json"
        data.json(filename, index=False)

    def saveAnalysis(self, data: pd.DataFrame) -> None:
        filename = f"analysis_{time.now()}" + str(uuid.uuid4()) + ".json"
        data.json(filename, index=False)

class DbService:
    """
    This class provides abstraction from the database
    """

    def __init__(self, strategy: DatabaseClientStrategy):
        self.strategy = strategy

    def save(self, data: pd.DataFrame) -> None:
        self.strategy.save(data)
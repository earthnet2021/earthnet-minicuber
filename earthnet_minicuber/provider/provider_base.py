
from abc import abstractmethod, ABC

class Provider(ABC):

    @abstractmethod
    def load_data(self, bbox, time_interval, **kwargs):
        pass

    
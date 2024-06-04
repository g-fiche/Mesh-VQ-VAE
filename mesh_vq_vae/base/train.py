from abc import ABC, abstractmethod, ABCMeta


class Train(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def one_epoch(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def load(self, path: str = "", optimizer: bool = True):
        pass

from abc import ABC, abstractmethod

# Abstract class for captioning


class Captioner(ABC):
    def __init__(self, device='cuda:0'):
        self.device = device

    @abstractmethod
    def _init_models(self):
        pass

    @abstractmethod
    def caption(self, imgs, user_prompt=None):
        pass

    @abstractmethod
    def stop(self):
        pass


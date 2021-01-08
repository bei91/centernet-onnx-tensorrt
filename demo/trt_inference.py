# -*-coding:utf-8-*-
from abc import ABCMeta, abstractmethod

import tensorrt as trt


class Singleton:
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self, *args, **kw):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls(*args, **kw)
        return self._instance[self._cls]


class TrtInference(metaclass=ABCMeta):
    TRT_LOGGER = trt.Logger()

    def __init__(self):
        self.image_height = None
        self.image_width = None

    @abstractmethod
    def load_engine(self): pass

    @abstractmethod
    def pre_process(self, image): pass

    @abstractmethod
    def infer(self, image, pre_results=None): pass

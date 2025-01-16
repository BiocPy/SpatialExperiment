from abc import ABC
import os

class SpatialImage(ABC):
    @classmethod
    def create(cls, x, is_url=None):
        if isinstance(x, SpatialImage):
            return x
        elif isinstance(x, str):
            if is_url is None:
                is_url = x.startswith(("http://", "https://", "ftp://"))
            if is_url:
                return RemoteSpatialImage(url=x)
            else:
                return StoredSpatialImage(path=x)
        else:
            raise ValueError("Unknown input type for 'x'")

class LoadedSpatialImage(SpatialImage):
    # use PIL (Pillow)?
    def __init__(self, image):
        self.image = image

class StoredSpatialImage(SpatialImage):
    def __init__(self, path):
        self.path = os.path.normpath(path)

class RemoteSpatialImage(SpatialImage):
    def __init__(self, url):
        self.url = url
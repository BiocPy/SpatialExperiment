import os

from PIL import Image

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


# TODO: add documentation, __repr__, __str__
class SpatialImage:
    def __init__(self, x):
        if isinstance(x, SpatialImage):
            self.image = x.image
            self.path = x.path
        elif isinstance(x, Image.Image):
            self.image = x
            self.path = None
        elif isinstance(x, str):
            if x.startswith(("http://", "https://", "ftp://")):
                raise ValueError("URLs are not supported for SpatialImage.")
            else:
                self.image = None
                self.path = os.path.normpath(x)
        else:
            raise ValueError("Unknown input type for 'x'")

    def load_image(self):
        """Load the image from the stored path into memory."""
        if self.image is None and self.path is not None:
            self.image = Image.open(self.path)
        return self.image

    def get_image(self):
        """Retrieve the image, loading it if necessary."""
        if self.image is None:
            return self.load_image()
        return self.image

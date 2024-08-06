import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw

class Cutout(object):
    """Randomly mask out one or more patches from an image with various shapes.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each side of the shape.
        shape (str): Shape of the patches to cut out ('triangle', 'circle', 'ellipse').
    """
    def __init__(self, n_holes, length, shape='triangle'):
        self.n_holes = n_holes
        self.length = length
        self.shape = shape

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of specified shape cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        shapes = ['triangle', 'circle', 'ellipse']
        if self.shape == 'random':
            self.shape = np.random.choice(shapes)
        elif self.shape not in shapes:
            raise ValueError(f"Shape must be one of {shapes}. Got {self.shape}.")

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            if self.shape == 'triangle':
                half_length = self.length // 2
                pts = np.array([[x, y - half_length],
                                [x - half_length, y + half_length],
                                [x + half_length, y + half_length]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 0)
            elif self.shape == 'circle':
                radius = self.length // 2
                cv2.circle(mask, (x, y), radius, 0, -1)
            elif self.shape == 'ellipse':
                axes = (self.length // 2, self.length // 4)  # Major and minor axes
                cv2.ellipse(mask, (x, y), axes, 0, 0, 360, 0, -1)

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

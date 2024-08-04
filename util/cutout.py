import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw

class Cutout(object):
    """Randomly mask out one or more patches from an image with various shapes.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each side of the shape.
        shape (str): Shape of the patches to cut out ('triangle', 'circle', 'star').
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
            elif self.shape == 'star':
                # Create a star mask with the size of the image
                star_mask = np.ones((h, w), np.float32)
                star_img = Image.new('L', (self.length, self.length), 0)
                draw = ImageDraw.Draw(star_img)
                draw.regular_polygon((self.length // 2, self.length // 2, self.length // 2), 5, fill=255)
                star_np = np.array(star_img)

                # Define the bounding box for the star mask
                y1 = max(0, y - self.length // 2)
                y2 = min(h, y + self.length // 2)
                x1 = max(0, x - self.length // 2)
                x2 = min(w, x + self.length // 2)

                # Place the star in the mask
                star_x1 = max(0, self.length // 2 - x)
                star_x2 = star_x1 + (x2 - x1)
                star_y1 = max(0, self.length // 2 - y)
                star_y2 = star_y1 + (y2 - y1)

                # Update the mask with the star shape
                star_mask[y1:y2, x1:x2] = np.clip(star_mask[y1:y2, x1:x2] - star_np[star_y1:star_y2, star_x1:star_x2], 0, 1)

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

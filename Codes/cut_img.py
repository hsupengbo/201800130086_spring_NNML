import math
from PIL import Image, ImageDraw
import numpy as np


class Rotate(object):

    def __init__(self, image: Image.Image, coordinate):
        self.image = image  # .convert('RGB')
        self.coordinate = coordinate
        self.xy = [tuple(self.coordinate[k]) for k in ['left_top', 'right_top', 'right_bottom', 'left_bottom']]
        self._mask = None
        self.image.putalpha(self.mask)

    @property
    def mask(self):
        if not self._mask:
            mask = Image.new('L', self.image.size, 0)
            draw = ImageDraw.Draw(mask, 'L')
            draw.polygon(self.xy, fill=255)
            self._mask = mask
        return self._mask

    def run(self):
        image = self.rotation_angle()
        box = image.getbbox()
        return image.crop(box).convert('RGB')

    def rotation_angle(self):
        x1, y1 = self.xy[0]
        x2, y2 = self.xy[1]
        angle = self.angle([x1, y1, x2, y2], [0, 0, 10, 0]) * -1
        return self.image.rotate(angle, expand=True)

    def angle(self, v1, v2):
        dx1 = v1[2] - v1[0]
        dy1 = v1[3] - v1[1]
        dx2 = v2[2] - v2[0]
        dy2 = v2[3] - v2[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180 / math.pi)
        angle2 = math.atan2(dy2, dx2)
        angle2 = int(angle2 * 180 / math.pi)
        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle


def get_img(image_path, image_name, x1, y1, x2, y2, x3, y3, x4, y4):  # ,new_path,,index
    image = Image.open(image_path + image_name)
    coordinate = {'left_top': [x1, y1], 'right_top': [x2, y2], 'right_bottom': [x4, y4], 'left_bottom': [x3, y3]}
    rotate = Rotate(image, coordinate)
    return rotate.run()  # .save(new_path+f"_{index}"+".jpg")


def get_rotate_img(img_text, x1, y1, x2, y2, x3, y3, x4, y4):  # ,new_path,,index
    image = Image.fromarray(img_text)
    coordinate = {'left_top': [x1, y1], 'right_top': [x2, y2], 'right_bottom': [x4, y4], 'left_bottom': [x3, y3]}
    rotate = Rotate(image, coordinate)
    img_text = np.array(rotate.run())
    return  img_text

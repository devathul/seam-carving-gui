# coding: utf-8

# by z0gSh1u @ 2019-08 (zx.cs@qq.com) | MIT LICENSE

import imageio
import numpy as np

import matplotlib.pyplot as plt

'''
  image coordinate system:
  (0, 0)
    ------------> y axis
    |
    |
    |
    v
  x axis
'''

class SeamCarver:

  def __init__(self, img):
    self._img = img

  def get_pixel(self, x, y):
    img = self._img
    if x < 0 or y < 0 or x >= img.shape[0] or y >= img.shape[1]:
      return 0  # performing zero-padding in fact
    return sum(img[x][y])  # r + g + b, flatten 3-ch to 1-d

  def calculate_energy_map(self):
    # might take a quite long time
    Sobel = [
      np.array([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
      ]),
      np.array([
        [1., 0., -1.],
        [2., 0., -2.],
        [1., 0., -1.]
      ])
    ] # Sobel operator
    img = self._img
    px = self.get_pixel  # alias
    energy_map = np.empty(img.shape[:-1])  # remove 3-ch shape
    max_energy = 0
    for x in range(img.shape[0]):
      for y in range(img.shape[1]):
        neighbour = np.array([
          [px(x - 1, y - 1), px(x - 1, y), px(x - 1, y + 1)],
          [px(x, y - 1), px(x, y), px(x, y + 1)],
          [px(x + 1, y - 1), px(x + 1, y), px(x + 1, y + 1)]
        ]).astype('float32')
        conv = lambda kernel: np.absolute(np.sum(np.multiply(neighbour, kernel)))
        energy_map[x][y] = conv(Sobel[0]) + conv(Sobel[1])
        max_energy = max(energy_map[x][y], max_energy)
    return energy_map, max_energy

  def calculate_seam(self, energy_map=None):
    if energy_map is None:
      energy_map = self.calculate_energy_map()
    # TODO: waiting for update

if __name__ == '__main__':
  im = imageio.imread('test.jpg')
  sc = SeamCarver(im)
  em = sc.calculate_energy_map()
  plt.imshow(em)
  plt.show()
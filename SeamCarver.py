# coding: utf-8

# by z0gSh1u @ 2019-08 (zx.cs@qq.com) | MIT LICENSE

import imageio
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

'''
  image coordinate system:
  (0, 0)
    ----> y axis
    |
    v
  x axis
'''

class SeamCarver:

  def __init__(self, img):
    self.img = img # r, c, 3-ch
    self.backtrack = None
    self.energy_map = None # 1-ch
    self.max_energy = None
    self.M = None

  def get_pixel(self, x, y):
    img = self.img
    if x < 0 or y < 0 or x >= img.shape[0] or y >= img.shape[1]:
      return 0  # performing zero-padding in fact
    return sum(img[x][y])  # r + g + b, flatten 3-ch to 1-d

  def calculate_energy_map(self):
    print("[calculate_energy_map]")
    # this function might take a quite long time
    Sobel = [
      np.array([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
      ]), # u
      np.array([
        [1., 0., -1.],
        [2., 0., -2.],
        [1., 0., -1.]
      ]) # v
    ] # Sobel operator
    img = self.img
    px = self.get_pixel  # alias
    energy_map = np.empty(img.shape[:-1])  # remove 3-ch shape
    max_energy = 0
    for x in trange(img.shape[0]):
      for y in range(img.shape[1]):
        neighbour = np.array([
          [px(x - 1, y - 1), px(x - 1, y), px(x - 1, y + 1)],
          [px(x, y - 1), px(x, y), px(x, y + 1)],
          [px(x + 1, y - 1), px(x + 1, y), px(x + 1, y + 1)]
        ]).astype('float32')
        conv = lambda kernel: np.absolute(np.sum(np.multiply(neighbour, kernel)))
        energy_map[x][y] = conv(Sobel[0]) + conv(Sobel[1])
        max_energy = max(energy_map[x][y], max_energy)
    self.energy_map, self.max_energy = energy_map, max_energy

  def calculate_seam(self):
    img = self.img
    r, c, _ = img.shape # row, col, channel
    M = self.energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)
    for i in range(1, r):
      for j in range(0, c):
        if j == 0: # col = 0
          idx = np.argmin(M[i - 1, j: j + 2]) # find rightward
          backtrack[i, j] = idx + j
          min_energy = M[i - 1, idx + j]
        else:
          idx = np.argmin(M[i - 1, j - 1: j + 2])
          backtrack[i, j] = idx + j - 1
          min_energy = M[i - 1, idx + j - 1]
        # finding seam by this method can ensure seam connects all points
        # in 8-neighbourhood
        M[i, j] += min_energy # dynamic programming, update energy so far
    self.M, self.backtrack = M, backtrack

  def carve_one_column(self):
    img = self.img
    r, c, _ = img.shape

    print(r, c, _)

    self.calculate_seam()
    M, backtrack = self.M, self.backtrack
    mask = np.ones((r, c), dtype=np.bool) # `True` means keep, `False` means carve

    j = np.argmin(M[-1]) # last row, minimum energy start point
    for i in reversed(range(r)):
      mask[i, j] = False
      j = backtrack[i, j]
    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, _))

    self.img = img

  def crop_column(self, target_scale):
    print("[crop_column]")
    img = self.img
    r, c, _ = img.shape
    target_c = int(target_scale * c)
    for i in trange(c - target_c):
      self.carve_one_column()

  def crop_row(self, target_scale):
    print("[crop_row]")
    img = self.img
    self.img = np.rot90(img, 1) # 90 degrees
    self.crop_column(target_scale)
    img = self.img
    self.img = np.rot90(img, 3) # 270 degrees

if __name__ == '__main__':
  im = imageio.imread('test.jpg')
  sc = SeamCarver(im)
  # sc.calculate_energy_map()
  # plt.imshow(sc.energy_map)
  # plt.show()
  sc.calculate_energy_map()
  sc.crop_column(0.7)
  plt.imshow(sc.img)
  plt.show()
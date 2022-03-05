# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2020'
__author__ = u'Lic. Manuel Aguado Martínez'


class LogGaborFilter():
    """A logarithmic Gabor filter"""

    def __init__(self, orient_map, freq_map, curv_map=None, mask=None,
                 bsize=16, sigma=8, wsize=64, sigma_orient=0.174,
                 sigma_radio=0.4):
        """Creates a Log Gabor Filter"""
        # Maps
        self.orient_map = orient_map
        self.freq_map = freq_map
        self.curv_map = curv_map
        self.mask = mask

        # Params
        self.bsize = bsize
        self.wsize = wsize
        self.sigma = sigma
        self.sigma_orient = sigma_orient
        self.sigma_radio = sigma_radio

        if self.curv_map is None:
            self.curv_map = np.zeros_like(self.orient_map)

        if self.mask is None:
            self.mask = np.ones_like(self.orient_map)

        self.gfilters = self._get_filters()
        self.wfilter = self._get_wind_filter()

    def _get_filters(self):
        """Creates the gabor filter"""

        blk_h, blk_w = self.freq_map.shape

        # Creating grid
        x, y = np.meshgrid(range(self.wsize), range(self.wsize))

        # Centered grid
        center = self.wsize // 2
        x_c, y_c = x - center, y - center

        # Grid radius
        radius = np.sqrt(x_c * x_c + y_c * y_c) / (self.wsize - 1)

        # Filters
        g_filters = [[None] * blk_w for _ in range(blk_h)]

        for i in range(blk_h):
            for j in range(blk_w):

                if not self.mask[i, j]:
                    continue

                freq = self.freq_map[i, j]
                ori = self.orient_map[i, j]
                curv = self.curv_map[i, j]

                # Creating log gabor filter
                freq_radius = radius / freq
                freq_radius[radius == 0] = 1
                log_gabor = (-0.5 * (np.log(freq_radius)) ** 2)
                log_gabor = log_gabor / (self.sigma_radio ** 2)
                log_gabor = np.exp(log_gabor)
                log_gabor[radius == 0] = 0

                # Rotating
                x_rot = x_c * np.cos(ori) + y_c * np.sin(ori)
                y_rot = y_c * np.cos(ori) - x_c * np.sin(ori)
                d_theta = np.arctan2(y_rot, x_rot)

                # Calculating spread
                spread = (-0.5 * d_theta ** 2)
                spread = spread / ((self.sigma_orient + curv) ** 2)
                spread = np.exp(spread)

                # Final filter
                amp = 1
                log_gabor = log_gabor * spread
                g_filters[i][j] = amp * log_gabor / np.max(log_gabor)

        return g_filters

    def _get_wind_filter(self):
        """Creates a filter to fusion windows"""
        # Creating grid
        x, y = np.meshgrid(range(self.wsize), range(self.wsize))

        # Centered grid
        center = self.wsize // 2
        x_c, y_c = x - center, y - center

        # Wind filter
        wfilter = np.exp(-0.5 * (x_c ** 2 + y_c ** 2) / self.sigma ** 2)

        return wfilter

    def apply(self, img):
        """Apply the filter over an image"""

        # Padding input image
        ovp_size = (self.wsize - self.bsize) // 2
        img = np.lib.pad(img, (ovp_size, ovp_size), 'symmetric')
        fimage = np.zeros_like(img, dtype=np.float64)

        for i in range(self.orient_map.shape[0]):
            for j in range(self.orient_map.shape[1]):

                if not self.mask[i, j]:
                    continue

                x0, x1 = i * self.bsize, i * self.bsize + self.wsize
                y0, y1 = j * self.bsize, j * self.bsize + self.wsize

                dwind = img[x0:x1, y0:y1]
                dwind = dwind * self.wfilter
                dwind = np.fft.fft2(dwind)
                dwind = np.fft.fftshift(dwind)
                dwind *= self.gfilters[i][j]
                dwind = np.fft.ifftshift(dwind)
                dwind = np.fft.ifft2(dwind).real

                fimage[x0:x1, y0:y1] += dwind

        fimage = fimage[ovp_size: -ovp_size, ovp_size: -ovp_size]

        minimum = np.min(fimage)
        maximum = np.max(fimage)
        fimage = (fimage - minimum) / (maximum - minimum) * 255

        thr = -minimum / (maximum - minimum) * 255

        return fimage, thr

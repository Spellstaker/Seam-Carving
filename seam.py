import numpy as np


class SeamCarve():
    __max_energy = 1000000.0

    def __init__(self, img):
        self.__arr = img.astype(int)
        self.__height, self.__width = img.shape[:2]
        self.__energy_arr = np.empty((self.__height, self.__width))
        self.__compute_energy_arr()

    def __is_border(self, i, j):
        return (i == 0 or i == self.__height - 1) or (j == 0 or j == self.__width - 1)

    def __compute_energy(self, i, j):
        if self.__is_border(i, j):
            return self.__max_energy

        b = abs(self.__arr[i - 1, j, 0] - self.__arr[i + 1, j, 0])
        g = abs(self.__arr[i - 1, j, 1] - self.__arr[i + 1, j, 1])
        r = abs(self.__arr[i - 1, j, 2] - self.__arr[i + 1, j, 2])

        b += abs(self.__arr[i, j - 1, 0] - self.__arr[i, j + 1, 0])
        g += abs(self.__arr[i, j - 1, 1] - self.__arr[i, j + 1, 1])
        r += abs(self.__arr[i, j - 1, 2] - self.__arr[i, j + 1, 2])

        energy = b + g + r

        return energy

    def __swapaxes(self):
        np.swapaxes(self.__energy_arr, 0, 1)
        np.swapaxes(self.__arr, 0, 1)
        self.__height, self.__width = self.__width, self.__height

    def __compute_energy_arr(self):
        self.__energy_arr[[0, -1], :] = self.__max_energy
        self.__energy_arr[:, [0, -1]] = self.__max_energy

        self.__energy_arr[1:-1, 1:-1] = np.add.reduce(
            np.abs(self.__arr[:-2, 1:-1] - self.__arr[2:, 1:-1]), -1)
        self.__energy_arr[1:-1, 1:-1] += np.add.reduce(
            np.abs(self.__arr[1:-1, :-2] - self.__arr[1:-1, 2:]), -1)

    def __compute_seam(self, horizontal=False):
        if horizontal:
            self.__swapaxes()

        energy_sum_arr = np.empty_like(self.__energy_arr)

        energy_sum_arr[0] = self.__energy_arr[0]
        for i in range(1, self.__height):
            energy_sum_arr[i, :-1] = np.minimum(
                energy_sum_arr[i - 1, :-1], energy_sum_arr[i - 1, 1:])
            energy_sum_arr[i, 1:] = np.minimum(
                energy_sum_arr[i, :-1], energy_sum_arr[i - 1, 1:])
            energy_sum_arr[i] += self.__energy_arr[i]

        seam = np.empty(self.__height, dtype=int)
        seam[-1] = np.argmin(energy_sum_arr[-1, :])
        seam_energy = energy_sum_arr[-1, seam[-1]]

        for i in range(self.__height - 2, -1, -1):
            l, r = max(0, seam[i + 1] -
                        1), min(seam[i + 1] + 2, self.__width)
            seam[i] = l + np.argmin(energy_sum_arr[i, l: r])

        if horizontal:
            self.__swapaxes()

        return (seam_energy, seam)

    def __carve(self, horizontal=True, seam=None, remove=True):
        if horizontal:
            self.__swapaxes()
        
        if seam is None:
            seam = self.__compute_seam()[1]
        if remove:
            self.__width -= 1
        else:
            self.__width += 1

        new_arr = np.empty((self.__height, self.__width, 3))
        new_energy_arr = np.empty((self.__height, self.__width))
        mp_deleted_count = 0

        for i, j in enumerate(seam):
            if remove:
                if self.__energy_arr[i, j] < 0:
                    mp_deleted_count += 1
                new_energy_arr[i] = np.delete(
                    self.__energy_arr[i], j)
                new_arr[i] = np.delete(self.__arr[i], j, 0)
            else:
                new_energy_arr[i] = np.insert(
                    self.__energy_arr[i], j, 0, 0)

                new_pixel = self.__arr[i, j]
                if not self.__is_border(i, j):
                    new_pixel = (
                        self.__arr[i, j - 1] + self.__arr[i, j + 1]) // 2

                new_arr[i] = np.insert(self.__arr[i], j, new_pixel, 0)

        self.__arr = new_arr
        self.__energy_arr = new_energy_arr

        for i, j in enumerate(seam):
            for k in range(j - 1, j + 1):
                if 0 <= k < self.__width and self.__energy_arr[i, k] >= 0:
                    self.__energy_arr[i, k] = self.__compute_energy(i, k)
        
        if horizontal:
            self.__swapaxes()

        return mp_deleted_count

    def rescale(self, new_height=None, new_width=None):
        if new_height is None:
            new_height = self.__height
        if new_width is None:
            new_width = self.__width

        while self.__width != new_width:
            self.__carve(horizontal=False, remove=self.__width > new_width)
        while self.__height != new_height:
            self.__carve(horizontal=True, remove=self.__height > new_height)

    def remove_mask(self, mask):
        mp_count = np.count_nonzero(mask)

        self.__energy_arr[mask] *= -(self.__max_energy ** 2)
        self.__energy_arr[mask] -= (self.__max_energy ** 2)

        while mp_count:
            v_seam_energy, v_seam = self.__compute_seam(True)
            h_seam_energy, h_seam = self.__compute_seam(False)

            horizontal, seam = False, v_seam

            if v_seam_energy > h_seam_energy:
                horizontal, seam = True, h_seam

            mp_count -= self.__carve(horizontal, seam)


    def image(self):
        return self.__arr.astype(np.uint8)

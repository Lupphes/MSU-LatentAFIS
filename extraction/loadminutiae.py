"""Load edited minutiae"""

import numpy as np


def load_edited_minutiae(mnt_file):
    """Load edited minutiae"""
    with open(mnt_file) as mf:
        lines = mf.readlines()
        count = int(lines[0])

        minutiae = np.zeros((count, 4))

        for i, l in enumerate(lines[1:]):
            minutiae[i] = np.array(list(map(float, l.split(' '))))

    return minutiae

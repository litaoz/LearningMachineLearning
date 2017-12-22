# Recipe3.py
#
# Demonstrate the importance of selecting good features using
# a toy example on breeds of dogs
# 
# Tutorial on using machine learning from Google Developers
# https://www.youtube.com/watch?v=N9fDIAflCMY
#
# Additional comments and notes written by LZ
# Updated 10/1/1
#

import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()

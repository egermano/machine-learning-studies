# What Makes a Good Feature? - Machine Learning Recipes #3
# https://www.youtube.com/watch?v=N9fDIAflCMY&index=3&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal

import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()

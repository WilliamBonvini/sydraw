import math
import random as rand
import numpy as np

# draws a random point belonging to a circle with center (h,k) and radius r
def point(h, k, r,noiseperc):
    theta = rand.random() * 2 * math.pi
    x = h + math.cos(theta) * r
    noise = np.random.normal(0, noiseperc)
    x = x + noise
    y = k + math.sin(theta) * r
    noise = np.random.normal(0, noiseperc)
    y = y + noise
    return x, y



# I could insert this function in the one above, but let's first write it independetly.
# this function creates outliers within a bounding box
# - whose length is 6 times the radius of the circle (mid point is the center of the circle)
# - whose height is 6 times the radius of the circle (mid point is the center of the circle)
def outliers_uniform(h, k, r):
    x = rand.uniform(h - 2 * r, h + 2 * r)
    y = rand.uniform(k - 2 * r, k + 2 * r)
    return x, y


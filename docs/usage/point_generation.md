# Point Generation

Import the necessary module from the `sydraw` package:

```python
from sydraw import synth
```
## Model: Circles

### Example: Generating Points from a Circle

This example demonstrates generating 10 randomly sampled points from a user-specified circle:

```python
circle = synth.circle(radius=3.0, center=(1.0,1.0), n=10, noise_perc=0.02, homogeneous=True)
```

This code generates a NumPy array of 2D points in homogeneous coordinates. The points are sampled from a circle with a radius of `3` and a center at coordinates `(1.0,1.0)`. The resulting array might look like this:

```
array([[-1.51779408, -0.63116921,  1.        ],
       [-1.83302435,  1.98690071,  1.        ],
       [-0.0106027 ,  3.82465612,  1.        ],
       [-0.32848561,  3.68981895,  1.        ],
       [-1.93858646,  0.39608805,  1.        ],
       [ 3.66809769, -0.37158839,  1.        ],
       [-0.54906204,  3.56912569,  1.        ],
       [ 3.88941592,  1.8070165 ,  1.        ],
       [ 2.56806854, -1.55756937,  1.        ],
       [ 3.88477601,  1.82344847,  1.        ]])
```

### Dataset Generation

To generate a (single or multi-model) dataset of circles:

```python
circle = synth.circles_dataset(nm=3, ns=400, n=1000, noise_perc=0.02, outliers_perc=0.20)
```

The code above returns a NumPy array with the shape `(ns, n, 3)`, where:
- `ns` is the number of samples (each sample is an array of data points).
- `n` is the number of points in each sample.
- The last axis of the array has a dimensionality of 3:  

  - The first two dimensions represent the x and y coordinates of the data points.
  - The third dimension contains the _class label_.

The **class labels** are defined as follows:
- `0`: Outlier.
- `1`, `2`, `3`, etc.: Model class identifier.

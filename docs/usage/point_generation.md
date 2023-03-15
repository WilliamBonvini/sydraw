
# Point Generation

```python
from sydraw import synth
```
### Example 

generate 10 randomly sampled points from a user-specified circle.
```python
circle = synth.circle(radius=3.0, center=(1.0,1.0), n=10, noise_perc=0.02, homogeneous=True)
```

The code above will generate a numpy array of 2D points in homogeneous coordinates.   
Such points belong to the circle with radius = 3 and center = (1.0,1.0).
```
>>> array([[-1.51779408, -0.63116921,  1.        ],
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


##### model coefficients generation
```python
circle = synth.circles_dataset(nm=3, ns=400, n=1000, noise_perc=0.02, outliers_perc=0.20)
```

The code above will return a numpy array with shape (ns, n, 3).  
Where the first two dimension are:

- ns: number of samples (a sample is defined as a point cloud)
- n: number of points in each sample

while the first two 
- nm: number of models in each sample

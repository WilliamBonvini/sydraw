# Syndalib
Syndalib (Synthetic Data Library) is a tool to create synthetic pointclouds for single/multi-model single/multi-class tasks.
I'll use the term "sample" to indicate a generic point cloud.
Syndalib gives you the possibility to set:
- the number of models within each sample
- total number of points withing each sample
- percentage of outliers within each sample
- class of each model ( i.e. you can choose to generate pointclouds, each with 2 models: a line and a circle)

The available classes of model so far are:
- Circles
- Ellipses
- Line
- Generic Conic (includes all the ones above, not implemented hyperbola yet)


# Setup

# syndalib
syndalib (Synthetic Data Library) is a library that helps you create synthetic 2D point clouds for single/multi-model single/multi-class tasks.  
I'll use the term "sample" to indicate a generic point cloud.  
It gives you the possibility to fix a vast set of hyperparameters for each class of models you are interested in generating.  
Models are saved in a .mat file format.  


# Setup





# Usage
You can generate models of circles, lines and ellipses.
the generation process is straight-forward and it is shown in the following code snippet

hyperparameters:
- the number of models within each sample
- total number of points withing each sample
- percentage of outliers within each sample
- class of each model ( i.e. you can choose to generate pointclouds, each with 2 models: a line and a circle)

The available classes of model so far are:
- Circles
- Ellipses
- Line
- Generic Conic (includes all the ones above, not implemented hyperbola yet)

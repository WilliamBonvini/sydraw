# syndalib
syndalib (Synthetic Data Library) is a library that helps you create synthetic 2D point clouds for single/multi-model single/multi-class tasks.  
It gives you the possibility to fix a vast set of hyperparameters for each class of models you are interested in generating.  
Models are saved in a .mat file format.  

# Setup





# Usage
You can generate models of circles, lines and ellipses.  
You can define a vast set of parameters to specify the sampling space and the characteristics of your models (the hyperparameters change for each model, but each of them consists in a interval of values the hyperparameter can take). In this README you'll find a section for each class of models in which I'll dig deeper into the hyperparameters I provide.  
the generation process is straight-forward and it is shown in the following code snippet

'''
from syndalib import syn2d

options = {
    "outliers": {
                "x_r": (-2.5, 2.5),
                "y_r": (-2.5, 2.5)
    },
     "circles": {
               "radius_r": (0.5, 1.5),
               "x_center_r": (-1.0, 1.0),
               "y_center_r": (-1.0, 1.0),
    },

    "lines": {
                "x_r": (-2.5, 2.5),
                "y_r": (-2.5, 2.5)
    },

    "ellipses": {
        "radius_r": (0.5, 1.5),
        "x_center_r": (-1, 1),
        "y_center_r": (-1, 1),
        "width_r": (0.1, 1),
        "height_r": (0.1, 1)
    },
}

syn2d.set_options(options)

'''




# Circles


# Ellipses

# Lines

# Conics

- Generic Conic (includes all the ones above, not implemented hyperbola yet)

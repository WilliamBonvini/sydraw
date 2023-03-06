# Sydraw

<h1 align="center">
<img src="https://github.com/WilliamBonvini/sydraw/raw/master/docs/media/imgs/sydraw.jpeg" width="150">
</h1><br>

[![PyPI License](https://img.shields.io/pypi/l/sydraw.svg)](https://pypi.org/project/sydraw)
[![PyPI Version](https://img.shields.io/pypi/v/sydraw.svg)](https://pypi.org/project/sydraw)
[![PyPI Downloads](https://img.shields.io/pypi/dm/sydraw.svg?color=orange)](https://pypistats.org/packages/sydraw)
![](https://img.shields.io/badge/contributions-welcome-green.svg)

Sydraw is a python library that helps you create synthetic 2D point clouds for single/multi-model single/multi-class tasks.  
It gives you the possibility to fix a set of hyper-parameters (i.e. outliers percentage, noise) for the parametric models you want to generate.  

|                                        Single Class - Single Model                                         |                                        Single Class -  Multi Model                                         |                                         Multi Class - Multi Model                                          |   
|:----------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/WilliamBonvini/sydraw/raw/master/docs/media/imgs/scsm1.png" style="zoom:50%"> | <img src="https://github.com/WilliamBonvini/sydraw/raw/master/docs/media/imgs/scmm1.png" style="zoom:50%"> | <img src="https://github.com/WilliamBonvini/sydraw/raw/master/docs/media/imgs/mcmm1.png" style="zoom:50%"> |
                        

## Setup

### Requirements

* Python 3.9+

### Installation

Install it directly into an activated virtual environment:

```text
$ pip install sydraw
```

or add it to your [Poetry](https://poetry.eustace.io/) project:

```text
$ poetry add sydraw
```

## Usage

After installation, the package can be imported:

```text
$ python
>>> import sydraw
>>> sydraw.__version__
```

Currently supported parametric models:
* circle
* ellipse
* line
* hyperbola

# Sydraw

<div align="center">
  <table>
    <tr>
      <td>
        <img src="https://github.com/WilliamBonvini/sydraw/raw/master/docs/media/imgs/featured.png" width="500" alt="Sydraw Logo">
      </td>
      <td>
        <strong>Sydraw</strong> is a Python library designed to assist in the creation of synthetic 2D point clouds, tailored for both single and multi-model fitting problems across single and multi-class scenarios. It offers the ability to finely tune hyper-parameters such as outlier percentages and noise levels, providing detailed control in the generation of parametric models.
      </td>
    </tr>
  </table>
</div>

<div align="center">
  <a href="https://pypi.org/project/sydraw">
    <img src="https://img.shields.io/pypi/l/sydraw.svg" alt="PyPI License">
  </a>
  <a href="https://pypi.org/project/sydraw">
    <img src="https://img.shields.io/pypi/v/sydraw.svg" alt="PyPI Version">
  </a>
  <a href="https://pypistats.org/packages/sydraw">
    <img src="https://img.shields.io/pypi/dm/sydraw.svg?color=orange" alt="PyPI Downloads">
  </a>
  <img src="https://img.shields.io/badge/contributions-welcome-green.svg" alt="Contributions Welcome">
</div>

---

### Examples of Generated Models

<div align="center">
  <table>
    <tr>
      <td>Single Class - Single Model</td>
      <td>Single Class - Multi Model</td>
      <td>Multi Class - Multi Model</td>
      <td rowspan="2">
        <a href="https://sydraw-demo.onrender.com/">
          <img src="https://github.com/WilliamBonvini/sydraw/raw/master/docs/media/imgs/demo.png" width="50" alt="Demo">
        </a>
      </td>
    </tr>
    <tr>
      <td>
        <img src="https://github.com/WilliamBonvini/sydraw/raw/master/docs/media/imgs/scsm1.png" alt="Single Class Single Model" width="200">
      </td>
      <td>
        <img src="https://github.com/WilliamBonvini/sydraw/raw/master/docs/media/imgs/scmm1.png" alt="Single Class Multi Model" width="200">
      </td>
      <td>
        <img src="https://github.com/WilliamBonvini/sydraw/raw/master/docs/media/imgs/mcmm1.png" alt="Multi Class Multi Model" width="200">
      </td>
    </tr>
  </table>
</div>

---

## Setup

### Requirements

- Python 3.9 or later

### Installation

```
pip install sydraw
```


## Usage

Once installed, you can import it and start using it:
```python
>> import sydraw
>> print(sydraw.__version__)
```

### Supported Parametric Models

- Circle
- Ellipse
- Line
- Hyperbola

For a detailed guide and additional functionalities, visit the [official documentation](https://sydraw.readthedocs.io/en/latest/).

---

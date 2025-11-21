# flowfusion
[![Static Badge](https://img.shields.io/badge/arXiv-2402.00935-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2402.00935)
[![Static Badge](https://img.shields.io/badge/arXiv-2406.19437-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2406.19437)
[![Static Badge](https://img.shields.io/badge/arXiv-2506.12122-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2506.12122)

Generative modelling and density estimation using diffusion models and flow-matching.

The code in this repository was developed as part of Alsing et al. ([2024](https://ui.adsabs.harvard.edu/abs/2024ApJS..274...12A/abstract)), and Thorp et al. ([2024](https://ui.adsabs.harvard.edu/abs/2024ApJ...975..145T/abstract), [2025](https://ui.adsabs.harvard.edu/abs/2025ApJ...993..240T/abstract)). In lieu of a more specific reference, please cite those papers if you make use of the code included here. Please also cite the papers associated with any dependencies of the code, particularly Chen et al. ([2018](https://ui.adsabs.harvard.edu/abs/2018arXiv180607366C/abstract)), which describes `torchdiffeq`.

# Installation
To install the code, please clone this repo:
```bash
  git clone https://github.com/Cosmo-Pop/flowfusion
```
Then move into the top level directory and run:
```bash
  pip install .
```
To install `flowfusion` without updating the dependencies:
```bash
pip install poetry
poetry install --no-update
```
This will obtain any dependencies and will install the code, which can then be imported in Python by doing:
```python
import flowfusion
```

# References
The code in this repository was developed and applied in the following papers:
- J. Alsing et al. (2024). ApJS 274, 12. [arXiv:2402.00935](https://arxiv.org/abs/2402.00935)
- S. Thorp et al. (2024). ApJ 975, 145. [arXiv:2406.19437](https://arxiv.org/abs/2406.19437)
- S. Thorp et al. (2025). ApJ, 993, 240. [arXiv:2506.12122](https://arxiv.org/abs/2506.12122)

For the mathematical underpinnings of the different modules within the code, please see (and consider citing) the following references, which our implementations largely follow:
### `flowfusion.diffusion`
- R.T.Q. Chen et al. (2018). NeurIPS 2018. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
- Y. Song et al. (2021a). ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
- Y. Song et al. (2021b). NeurIPS 2021. [arXiv:2101.09258](https://arxiv.org/abs/2101.09258)

### `flowfusion.flow`
- Y. Lipman et al. (2023). ICLR 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)

### `flowfusion.symplectic`
- P. Toth et al. (2020). ICLR 2020. [arXiv:1909.13789](https://arxiv.org/abs/1909.13789)

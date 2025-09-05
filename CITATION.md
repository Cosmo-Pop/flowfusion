# Citing flowfusion
If you make use of flowfusion, please cite the papers where the codebase was introduced, and the relevant dependencies. Below is a string of LaTeX code that could be used in, e.g., a software acknowledgements section.
```latex
\texttt{flowfusion} \citep{alsing24, thorp24, thorp25};
\texttt{numpy} \citep{harris20};
\texttt{torch} \citep{paszke19};
\texttt{torchdiffeq} \citep{chen18};
\texttt{tqdm} \citep{dacostaluis24}
```

We would also encourage users to acknowledge the theoretical works that our code is based on. These are listed below for the three main submodules of flowfusion.
```latex
\texttt{flowfusion.diffusion} \citep{chen18, song21_iclr, song21_neurips};
\texttt{flowfusion.flow} \citep{lipman23};
\texttt{flowfusion.symplectic_flow} \citep{toth20}.
```

BibTeX entries for all of these references are included below, based on NASA ADS and DBLP.
```bibtex
@ARTICLE{alsing24,
       author = {{Alsing}, Justin and {Thorp}, Stephen and {Deger}, Sinan and {Peiris}, Hiranya V. and {Leistedt}, Boris and {Mortlock}, Daniel and {Leja}, Joel},
        title = "{pop-cosmos: A Comprehensive Picture of the Galaxy Population from COSMOS Data}",
      journal = {\apjs},
     keywords = {Galaxy evolution, Galaxy abundances, Galaxy chemical evolution, Cosmological parameters, Cosmology, Redshift surveys, 594, 574, 580, 339, 343, 1378, Astrophysics - Astrophysics of Galaxies, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2024,
        month = sep,
       volume = {274},
       number = {1},
          eid = {12},
        pages = {12},
          doi = {10.3847/1538-4365/ad5c69},
archivePrefix = {arXiv},
       eprint = {2402.00935},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJS..274...12A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@INPROCEEDINGS{chen18,
       author = {Chen, Ricky T. Q. and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David K},
    booktitle = {Advances in Neural Information Processing Systems},
       editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
        pages = {6572--6583},
    publisher = {Curran Associates, Inc.},
        title = {Neural Ordinary Differential Equations},
       volume = {31},
         year = {2018},
archivePrefix = {arXiv},
       eprint = {1806.07366},
          url = {https://proceedings.neurips.cc/paper_files/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf}
}

@MISC{dacostaluis24,
       author = {{da Costa-Luis}, Casper and {Larroque}, Stephen Karl and {Altendorf}, Kyle and {Mary}, Hadrien and {richardsheridan} and {Korobov}, Mikhail and {Yorav-Raphael}, Noam and {Ivanov}, Ivan and {Bargull}, Marcel and {Rodrigues}, Nishant and {Shawn} and {Dektyarev}, Mikhail and {G{\'o}rny}, Micha{\l} and {mjstevens777} and {Pagel}, Matthew D. and {Zugnoni}, Martin and {JC} and {CrazyPython} and {Newey}, Charles and {Lee}, Antony and {pgajdos} and {Todd} and {Malmgren}, Staffan and {redbug312} and {Desh}, Orivej and {Nechaev}, Nikolay and {Boyle}, Mike and {Nordlund}, Max and {MapleCCC} and {McCracken}, Jack},
        title = "{tqdm: A fast, Extensible Progress Bar for Python and CLI}",
         year = 2024,
        month = nov,
          eid = {10.5281/zenodo.14231923},
          doi = {10.5281/zenodo.14231923},
      version = {v4.67.1},
 howpublished = {Zenodo},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024zndo..14231923D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{harris20,
       author = {{Harris}, Charles R. and {Millman}, K. Jarrod and {van der Walt}, St{\'e}fan J. and {Gommers}, Ralf and {Virtanen}, Pauli and {Cournapeau}, David and {Wieser}, Eric and {Taylor}, Julian and {Berg}, Sebastian and {Smith}, Nathaniel J. and {Kern}, Robert and {Picus}, Matti and {Hoyer}, Stephan and {van Kerkwijk}, Marten H. and {Brett}, Matthew and {Haldane}, Allan and {del R{\'\i}o}, Jaime Fern{\'a}ndez and {Wiebe}, Mark and {Peterson}, Pearu and {G{\'e}rard-Marchant}, Pierre and {Sheppard}, Kevin and {Reddy}, Tyler and {Weckesser}, Warren and {Abbasi}, Hameer and {Gohlke}, Christoph and {Oliphant}, Travis E.},
        title = "{Array programming with NumPy}",
      journal = {\nat},
     keywords = {Computer Science - Mathematical Software, Statistics - Computation},
         year = 2020,
        month = sep,
       volume = {585},
       number = {7825},
        pages = {357-362},
          doi = {10.1038/s41586-020-2649-2},
archivePrefix = {arXiv},
       eprint = {2006.10256},
 primaryClass = {cs.MS},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020Natur.585..357H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@INPROCEEDINGS{lipman23,
       author = {{Lipman}, Yaron and {Chen}, Ricky T.~Q. and {Ben-Hamu}, Heli and {Nickel}, Maximilian and {Le}, Matt},
        title = "{Flow Matching for Generative Modeling}",
    booktitle = {11th International Conference on Learning Representations},
         year = {2023},
          eid = {arXiv:2210.02747},
archivePrefix = {arXiv},
       eprint = {2210.02747},
 primaryClass = {cs.LG},
          url = {https://openreview.net/forum?id=PqvMRDCJT9t},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv221002747L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@INPROCEEDINGS{paszke19,
       author = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
    booktitle = {Advances in Neural Information Processing Systems},
       editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
        pages = {8024--8035},
    publisher = {Curran Associates, Inc.},
        title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
       volume = {32},
         year = {2019},
archivePrefix = {arXiv},
       eprint = {1912.01703},
          url = {https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf}
}

@INPROCEEDINGS{song21_iclr,
       author = {{Song}, Yang and {Sohl-Dickstein}, Jascha and {Kingma}, Diederik P. and {Kumar}, Abhishek and {Ermon}, Stefano and {Poole}, Ben},
        title = "{Score-Based Generative Modeling through Stochastic Differential Equations}",
    booktitle = {9th International Conference on Learning Representations},
         year = {2021},
          eid = {arXiv:2011.13456},
archivePrefix = {arXiv},
       eprint = {2011.13456},
 primaryClass = {cs.LG},
          url = {https://openreview.net/forum?id=PxTIG12RRHS},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv201113456S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@INPROCEEDINGS{song21_neurips,
       author = {{Song}, Yang and {Durkan}, Conor and {Murray}, Iain and {Ermon}, Stefano},
        title = "{Maximum Likelihood Training of Score-Based Diffusion Models}",
         year = 2021,
        month = jan,
archivePrefix = {arXiv},
       eprint = {2101.09258},
 primaryClass = {stat.ML},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210109258S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System},
       editor = {Marc'Aurelio Ranzato and Alina Beygelzimer and Yann N. Dauphin and Percy Liang and Jennifer Wortman Vaughan},
    booktitle = {Advances in Neural Information Processing Systems},
       volume = {34},
        pages = {1415--1428},
          url = {https://papers.nips.cc/paper/2021/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf}
}

@ARTICLE{thorp24,
       author = {{Thorp}, Stephen and {Alsing}, Justin and {Peiris}, Hiranya V. and {Deger}, Sinan and {Mortlock}, Daniel J. and {Leistedt}, Boris and {Leja}, Joel and {Loureiro}, Arthur},
        title = "{pop-cosmos: Scaleable Inference of Galaxy Properties and Redshifts with a Data-driven Population Model}",
      journal = {\apj},
     keywords = {Astrostatistics techniques, Redshift surveys, Galaxy photometry, Bayesian statistics, Affine invariant, Spectral energy distribution, 1886, 1378, 611, 1900, 1890, 2129, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2024,
        month = nov,
       volume = {975},
       number = {1},
          eid = {145},
        pages = {145},
          doi = {10.3847/1538-4357/ad7736},
archivePrefix = {arXiv},
       eprint = {2406.19437},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJ...975..145T},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{thorp25,
       author = {{Thorp}, Stephen and {Peiris}, Hiranya V. and {Jagwani}, Gurjeet and {Deger}, Sinan and {Alsing}, Justin and {Leistedt}, Boris and {Mortlock}, Daniel J. and {Halder}, Anik and {Leja}, Joel},
        title = "{pop-cosmos: Insights from generative modeling of a deep, infrared-selected galaxy population}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics of Galaxies, Cosmology and Nongalactic Astrophysics},
         year = 2025,
        month = jun,
          eid = {arXiv:2506.12122},
        pages = {arXiv:2506.12122},
          doi = {10.48550/arXiv.2506.12122},
archivePrefix = {arXiv},
       eprint = {2506.12122},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250612122T},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@INPROCEEDINGS{toth20,
       author = {{Toth}, Peter and {Jimenez Rezende}, Danilo and {Jaegle}, Andrew and {Racani{\`e}re}, S{\'e}bastien and {Botev}, Aleksandar and {Higgins}, Irina},
        title = "{Hamiltonian Generative Networks}",
    booktitle = {8th International Conference on Learning Representations},
         year = 2020,
          eid = {arXiv:1909.13789},
archivePrefix = {arXiv},
       eprint = {1909.13789},
 primaryClass = {cs.LG},
 		  url = {https://openreview.net/forum?id=HJenn6VFvB},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190913789T},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

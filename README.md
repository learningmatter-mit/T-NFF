# Temperature transferable Neural Force Field (TNFF) for coarse-grained molecular dynamics simulations

Implementation of the TNFF from our paper https://arxiv.org/abs/2007.14144

The TNFF code is based on SchNet [1-4], DimeNet [5]. It provides an interface to train and evaluate neural networks for force fields, this model includes two interaction blocks to better learn the many body Potential of Mean Force of a coarse-grained system. The system of example is an ionic liquid one.

This code repository is developed in the Learning Matter Lab (led by prof. Rafael Gomez-Bombarelli) at MIT. Please do not distribute.

## Installation from source

This software requires the following packages:

- [scikit-learn=0.23.1](http://scikit-learn.org/stable/)
- [PyTorch=1.4](http://pytorch.org)
- [ase=3.19.1](https://wiki.fysik.dtu.dk/ase/)
- [pandas=1.0.5](https://pandas.pydata.org/)
- [networkx=2.4](https://networkx.github.io/)
- [pymatgen=2020.7.3](https://pymatgen.org/)
- [sympy=1.6.1](https://www.sympy.org/)

We highly recommend to create a `conda` environment to run the code. To do that, use the following commands:

```bash
conda upgrade conda
conda create -n nff python=3.7 scikit-learn pytorch\>=1.2.0 cudatoolkit=10.0 ase pandas pymatgen sympy -c pytorch -c conda-forge
```

You need to activate the `nff` environment to install the NFF package:

```bash
conda activate nff
```

Finally, install the `nff` package by running:

```bash
pip install .
```

## Usage

### Usage with Jupyter Notebooks and other scripts

A series of tutorials illustrating how `nff` can be used in conjunction with Jupyter Notebooks or other scripts is provided in the `examples/` folder. Furthermore, it includes the necessary 


## References

* [1] K.T. Schütt. F. Arbabzadah. S. Chmiela, K.-R. Müller, A. Tkatchenko.  
*Quantum-chemical insights from deep tensor neural networks.*
Nature Communications **8**. 13890 (2017)   
[10.1038/ncomms13890](http://dx.doi.org/10.1038/ncomms13890)

* [2] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.  
*SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.*
Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017) [link](http://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions)

* [3] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.  
*SchNet - a deep learning architecture for molecules and materials.* 
The Journal of Chemical Physics 148(24), 241722 (2018) [10.1063/1.5019779](https://doi.org/10.1063/1.5019779)

* [4] K.T. Schütt, P. Kessel, M. Gastegger, K. Nicoli, A. Tkatchenko, K.-R. Müller.
*SchNetPack: A Deep Learning Toolbox For Atomistic Systems.*
J. Chem. Theory Comput. **15**(1), 448-455 (2019). [10.1021/acs.jctc.8b00908](https://doi.org/10.1021/acs.jctc.8b00908)

* [5] J. Klicpera, G. Janek, S. Günnemann. *Directional message passing for molecular graphs.* ICLR (2020). [URL](https://openreview.net/attachment?id=B1eWbxStPH&name=original_pdf).

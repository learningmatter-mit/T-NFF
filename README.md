# Temperature transferable Neural Force Field (TNFF) for coarse-grained molecular dynamics simulations

Implementation of the temperature transferable neural force field from our paper https://doi.org/10.1063/5.0022431

<p align="center">
  <img src="images/tnff.jpg" width="500">
</p>

The model is based on SchNet [1]. It provides an interface to train and evaluate neural networks for force fields, specifically tested on coarse-grained ionic liquid simulations. Furthermore, the work pipeline uses the coarse-graining auto-encoders for determining the statistically most viable coarse-grain mapping [2].

## Usage

Three notebooks run through the workflow of the paper

 - Part1_cg_mapping.ipynb
  Uses [coarse-grained auto-encoders](https://github.com/learningmatter-mit/Coarse-Graining-Auto-encoders) for determening the best mapping for the ionic liquid. Uses MD data from LAMMPS with a [force field for ionic liquids](https://github.com/agiliopadua/ilff). The full data set used for training can be found [3]
 - Part2_data_file_creation.ipynb
  Using the previously mentioned data and the newly generated coarse-grained mapping preparing the data for training, applying the coarse-grained filter for the training data.
 - Part3_temp_transfer.ipynb
  Training the model and running MD simulations on ASE. The hyperparameters in the model are the ones from the best run, though the data is not the full dataset. The best model from the paper is included in the ./examples/models/ directory

## Installation from source

This software requires the following packages:

- [PyTorch=1.4](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [ase=3.18](https://wiki.fysik.dtu.dk/ase/)
- [networkx=2.3](https://networkx.github.io/)

We highly recommend to create a `conda` environment to run the code. To do that, use the following commands:

```bash
conda upgrade conda
conda create -n nff python=3.7 scikit-learn pytorch>=1.2.0 cudatoolkit=10.0 ase pandas pymatgen -c pytorch -c conda-forge
```

You need to activate the `tnff` environment to install the NFF package:

```bash
conda activate nff
```

Finally, install the `tnff` package by running:

```bash
pip install .
```


## References

* [1] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.  
*SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.*
Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017) [link](http://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions)

* [2] Wang, W., & Gómez-Bombarelli, R.
*Coarse-graining auto-encoders for molecular dynamics.* npj Computational Materials, 5(1), 1-9 (2019) [link](https://www.nature.com/articles/s41524-019-0261-5)

* [3] Ruza, Jurgis (2021): MD simulation data for training the T-NFF model. figshare. Dataset. https://doi.org/10.6084/m9.figshare.12967217.v1 

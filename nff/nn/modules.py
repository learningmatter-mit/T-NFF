import numpy as np

import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ModuleDict

from nff.nn.layers import Dense, GaussianSmearing
from nff.utils.scatter import scatter_add
from nff.nn.activations import shifted_softplus
from nff.nn.graphconv import (
    MessagePassingModule,
    EdgeUpdateModule
)
from nff.nn.utils import construct_sequential, construct_module_dict
from nff.utils.scatter import compute_grad

import unittest
import itertools
import copy
import pdb




DEFAULT_BONDPRIOR_PARAM = {"k": 20.0,
                           'dif_bond_len': False}



class SchNetConv(MessagePassingModule):

    """The convolution layer with filter.
    
    Attributes:
        moduledict (TYPE): Description
    """

    def __init__(
        self,
        n_atom_basis,
        n_filters,
        n_gaussians,
        cutoff,
        trainable_gauss,
        dropout_rate,
    ):
        super(SchNetConv, self).__init__()
        self.moduledict = ModuleDict(
            {
                "message_edge_filter": Sequential(
                    GaussianSmearing(
                        start=0.0,
                        stop=cutoff,
                        n_gaussians=n_gaussians,
                        trainable=trainable_gauss,
                    ),
                    Dense(
                        in_features=n_gaussians,
                        out_features=n_gaussians,
                        dropout_rate=dropout_rate,
                    ),
                    shifted_softplus(),
                    Dense(
                        in_features=n_gaussians,
                        out_features=n_filters,
                        dropout_rate=dropout_rate,
                    ),
                ),
                "message_node_filter": Dense(
                    in_features=n_atom_basis,
                    out_features=n_filters,
                    dropout_rate=dropout_rate,
                ),
                "update_function": Sequential(
                    Dense(
                        in_features=n_filters,
                        out_features=n_atom_basis,
                        dropout_rate=dropout_rate,
                    ),
                    shifted_softplus(),
                    Dense(
                        in_features=n_atom_basis,
                        out_features=n_atom_basis,
                        dropout_rate=dropout_rate,
                    ),
                ),
            }
        )

    def message(self, r, e, a, aggr_wgt=None):
        """The message function for SchNet convoltuions 
        Args:
            r (TYPE): node inputs
            e (TYPE): edge inputs
            a (TYPE): neighbor list
            aggr_wgt (None, optional): Description

        Returns:
            TYPE: message should a pair of message and
        """
        # update edge feature
        e = self.moduledict["message_edge_filter"](e)
        # convection: update
        r = self.moduledict["message_node_filter"](r)

        # soft aggr if aggr_wght is provided
        if aggr_wgt is not None:
            r = r * aggr_wgt

        # combine node and edge info
        message = r[a[:, 0]] * e, r[a[:, 1]] * e  # (ri [] eij) -> rj, []: *, +, (,)
        return message

    def update(self, r):
        return self.moduledict["update_function"](r)


class NodeMultiTaskReadOut(nn.Module):
    """Stack Multi Task outputs

        example multitaskdict:

        multitaskdict = {
            'myenergy_0': [
                {'name': 'linear', 'param' : { 'in_features': 5, 'out_features': 20}},
                {'name': 'linear', 'param' : { 'in_features': 20, 'out_features': 1}}
            ],
            'myenergy_1': [
                {'name': 'linear', 'param' : { 'in_features': 5, 'out_features': 20}},
                {'name': 'linear', 'param' : { 'in_features': 20, 'out_features': 1}}
            ],
            'muliken_charges': [
                {'name': 'linear', 'param' : { 'in_features': 5, 'out_features': 20}},
                {'name': 'linear', 'param' : { 'in_features': 20, 'out_features': 1}}
            ]
        }

        example post_readout:

        def post_readout(predict_dict, readoutdict):
            sorted_keys = sorted(list(readoutdict.keys()))
            sorted_ens = torch.sort(torch.stack([predict_dict[key] for key in sorted_keys]))[0]
            sorted_dic = {key: val for key, val in zip(sorted_keys, sorted_ens) }
            return sorted_dic
    """

    def __init__(self, multitaskdict, post_readout=None):
        """Summary

        Args:
            multitaskdict (dict): dictionary that contains model information
        """
        super(NodeMultiTaskReadOut, self).__init__()
        self.readout = construct_module_dict(multitaskdict)
        self.post_readout = post_readout
        self.multitaskdict = multitaskdict

    def forward(self, r):
        predict_dict = dict()
        for key in self.readout:
            predict_dict[key] = self.readout[key](r)
        if self.post_readout is not None:
            predict_dict = self.post_readout(predict_dict, self.multitaskdict)

        return predict_dict



class BondPrior(torch.nn.Module):

    def __init__(self, modelparams=DEFAULT_BONDPRIOR_PARAM):
        torch.nn.Module.__init__(self)
        self.k = modelparams['k']
        self.dif_bond_len = modelparams.get('dif_bond_len', None)      
 
    def forward(self, batch):
        
        result = {}
        
        num_bonds = batch["num_bonds"].tolist()
        
        xyz = batch['nxyz'][:, 1:4]
        xyz.requires_grad = True
        bond_list = batch["bonds"]
        
        r_0 = batch['bond_len'].squeeze()
        
        r = (xyz[bond_list[:, 0]] - xyz[bond_list[:, 1]]).pow(2).sum(-1).sqrt()
       
        if self.dif_bond_len:
            r = torch.stack([r for r in torch.split(r, num_bonds[0])])
 
        e = self.k * ( r - r_0).pow(2)
        
        if self.dif_bond_len:
            E = e.sum(1)
        else:
            E = torch.stack([e.sum(0) for e in torch.split(e, num_bonds[0])])
        
        result['energy'] = E.sum().reshape(1,1)
        result['energy_grad'] = compute_grad(inputs=xyz, output=E)
        
        return result

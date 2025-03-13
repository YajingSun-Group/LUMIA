#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   plot_funs.py
@Time    :   2024/04/22 21:10:46
@Author  :   Zhang Qian
@Contact :   zhangqian.allen@gmail.com
@License :   Copyright (c) 2024 by Zhang Qian, All Rights Reserved. 
@Desc    :   None
"""

# here put the import lib
from rdkit import Chem, Geometry
import numpy as np
import matplotlib.pyplot as plt


def plot_fill_a_ring(mol,ring_atoms,ring_bonds,color_id = 1):
    
    colors = [
    (100, 100, 100), # 灰色，用于掩码用的颜色
    (86,180,233), # blue
    (230,159,0), # yellow
    (0,190,150), # green
    (204,121,167), # pale rose
    (180,141,255), # purple
    (254,46,152), # rose
    (254,97,0), # orange
    (120,94,240), # purple
    (100,143,255), # royal blue
    (213,94,0), # brown
    (0,114,178), # dark blue 
    (240,228,66), # light yellow
    (204,121,167), # pale rose
    (255,176,0) # yellow orange
]

    for i,x in enumerate(colors): # 归一化
        colors[i] = tuple(y/255 for y in x)
        
    color = colors[color_id]

    dict_a_colors = {int(idx):[color] for idx in ring_atoms}
    dict_b_colors = {int(idx):[color] for idx in ring_bonds}
    
    atomrads = {i:0.4 for i in dict_a_colors}
    widthmults = {i:4 for i in dict_b_colors}
    
    Chem.rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    
    d2d = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(500,500)
    dos = d2d.drawOptions()
    dos.useBWAtomPalette()
    dos.bondLineWidth = 3
    dos.baseFontSize=0.8
    
    d2d.DrawMoleculeWithHighlights(mol,
                                "",
                                dict_a_colors,
                                dict_b_colors,
                                atomrads, 
                                widthmults)
    
    
    d2d.ClearDrawing()
    
    ps = []
    for aidx in ring_atoms:
        pos = Geometry.Point2D(conf.GetAtomPosition(aidx))
        ps.append(pos)
    d2d.SetFillPolys(True)
    d2d.SetColour(color)
    d2d.DrawPolygon(ps)
    
    dos.clearBackground = False
    d2d.SetFontSize(22)
    
    
    d2d.DrawMoleculeWithHighlights(mol,"", dict_a_colors,
                                            dict_b_colors,
                                            atomrads, widthmults)
    
    d2d.FinishDrawing()
    
    svg = d2d.GetDrawingText()

    return svg



def plot_with_color(mol, atom_list,bond_list=None, lineWidth=2, atomrads=0.4,widthmults=2,fontSize=16,figsize=(500,500), color_id=1):
    # Thanks for MolSHAP: https://github.com/tiantz17/MolSHAP/blob/master/utils.py
    
    if bond_list is None:
        bond_list = []
        # get bond list based on atom_list and mol
        for bond in mol.GetBonds():
            if bond.GetBeginAtom().GetIdx() in atom_list and bond.GetEndAtom().GetIdx() in atom_list:
                bond_list.append(bond.GetIdx())
    
    colors = [
        (100, 100, 100), # gray
        (86,180,233), # blue
        (230,159,0), # yellow
        (0,190,150), # green
        (204,121,167), # pale rose
        (180,141,255), # purple
        (254,46,152), # rose
        (254,97,0), # orange
        (120,94,240), # purple
        (100,143,255), # royal blue
        (213,94,0), # brown
        (0,114,178), # dark blue 
        (240,228,66), # light yellow
        (204,121,167), # pale rose
        (255,176,0) # yellow orange
    ]

    # for i,x in enumerate(colors):
    #     colors[i] = tuple(y/255 for y in x)
    
    color = tuple(y/255 for y in colors[color_id])
    
    
    Chem.GetSSSR(mol)
    # 查找atom_list中的环，并保存到列表里
    rinfo  = mol.GetRingInfo()
    
    list_r = []    
    for aring in rinfo.AtomRings():
        overlap = np.intersect1d(aring, atom_list)
        if len(overlap) == len(aring):
            list_r.append(aring)
    
    # list_bond_idx = [[bond.GetIdx(), bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()] for bond in mol.GetBonds()]
    dict_color = {int(idx):[color] for i in range(len(list_r)) for idx in list_r[i]}
    # list_b = [[int(b[0]) for b in list_bond_idx if b[1] in list_r[i] and b[2] in list_r[i]] for i in range(len(list_r))]
    list_b = bond_list
    dict_b_color = {int(b):[color] for b in list_b}
    atomrads_ring = {i:atomrads for i in dict_color}
    widthmults_ring = {i:widthmults for i in dict_b_color}

    # Chem.rdDepictor.SetPreferCoordGen(True)
    Chem.rdDepictor.Compute2DCoords(mol)
    
    conf = mol.GetConformer()
    rinfo = mol.GetRingInfo()
    
    d2d = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(figsize[0],figsize[1])
    dos = d2d.drawOptions()
    dos.useBWAtomPalette()
    dos.bondLineWidth = lineWidth
    dos.minFontSize = fontSize
    # dos.setFontSize(fontSize)

    d2d.DrawMoleculeWithHighlights(mol,
                                   "",
                                   dict_color,
                                   dict_b_color,
                                   atomrads_ring, 
                                   widthmults_ring
                                   )
    d2d.ClearDrawing()
    for aring in list_r:
        ps = []
        for aidx in aring:
            pos = Geometry.Point2D(conf.GetAtomPosition(aidx))
            ps.append(pos)
        d2d.SetFillPolys(True)
        d2d.SetColour(color)
        d2d.DrawPolygon(ps)
    dos.clearBackground = False
    # d2d.SetFontSize(fontSize)

    #----------------------
    # now draw the molecule, with highlights:
    dict_color_all = {int(idx):[color] for idx in atom_list}
    dict_b_color_all = {int(b):[color] for b in bond_list}
    atomrads_all = {i:atomrads for i in dict_color_all}
    widthmults_all = {i:widthmults for i in dict_b_color_all}
    
    # d2d.DrawMoleculeWithHighlights(mol,"",dict_color_all,
    #                                        dict_b_color_all,
    #                                        atomrads_all, widthmults_all)
    
    d2d.DrawMoleculeWithHighlights(mol,"",dict_color_all,
                                           dict_b_color_all,
                                           atomrads_all, widthmults_all)
    d2d.FinishDrawing()
    pic = d2d.GetDrawingText()
    return pic    


def plot_with_colorful(mol, atom_list, bond_list=None, lineWidth=2, atomrads=0.4,widthmults=2,fontSize=16,figsize=(500,500), 
                    #    color_id=1
                       ):
    # Thanks for MolSHAP: https://github.com/tiantz17/MolSHAP/blob/master/utils.py
    
    if bond_list is None:
        bond_list = []
        # get bond list based on atom_list and mol
        for bond in mol.GetBonds():
            if bond.GetBeginAtom().GetIdx() in atom_list and bond.GetEndAtom().GetIdx() in atom_list:
                bond_list.append(bond.GetIdx())
    
    colors = [
        # (100, 100, 100), # gray
        (86,180,233), # blue
        (230,159,0), # yellow
        (0,190,150), # green
        (204,121,167), # pale rose
        (180,141,255), # purple
        (254,46,152), # rose
        (254,97,0), # orange
        (120,94,240), # purple
        (100,143,255), # royal blue
        (213,94,0), # brown
        (0,114,178), # dark blue 
        (240,228,66), # light yellow
        (204,121,167), # pale rose
        (255,176,0) # yellow orange
    ]

    for i,x in enumerate(colors):
        colors[i] = tuple(y/255 for y in x)
    
    # color = tuple(y/255 for y in colors[color_id])
    
    
    Chem.GetSSSR(mol)
    # 查找atom_list中的环，并保存到列表里
    rinfo  = mol.GetRingInfo()
    
    list_r = []    
    for aring in rinfo.AtomRings():
        overlap = np.intersect1d(aring, atom_list)
        if len(overlap) == len(aring):
            list_r.append(aring)
    
    # list_bond_idx = [[bond.GetIdx(), bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()] for bond in mol.GetBonds()]
    dict_color = {int(idx):[colors[i%len(colors)]] for i in range(len(list_r)) for idx in list_r[i]}
    # list_b = [[int(b[0]) for b in list_bond_idx if b[1] in list_r[i] and b[2] in list_r[i]] for i in range(len(list_r))]
    list_b = bond_list
    dict_b_color = {int(b):[colors[i%len(colors)]] for b in list_b}
    atomrads_ring = {i:atomrads for i in dict_color}
    widthmults_ring = {i:widthmults for i in dict_b_color}

    # Chem.rdDepictor.SetPreferCoordGen(True)
    Chem.rdDepictor.Compute2DCoords(mol)
    
    conf = mol.GetConformer()
    rinfo = mol.GetRingInfo()
    
    d2d = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(figsize[0],figsize[1])
    dos = d2d.drawOptions()
    dos.useBWAtomPalette()
    dos.bondLineWidth = lineWidth
    dos.minFontSize = fontSize
    # dos.setFontSize(fontSize)
    
    rings = []
    for i,aring in enumerate(list_r):
        rings.append([aring, colors[i%len(colors)]])

    
    d2d.DrawMoleculeWithHighlights(mol,
                                   "",
                                   dict_color,
                                   dict_b_color,
                                   atomrads_ring, 
                                   widthmults_ring
                                   )
    
    d2d.ClearDrawing()
    for (aring,color) in rings:
        ps = []
        for aidx in aring:
            pos = Geometry.Point2D(conf.GetAtomPosition(aidx))
            ps.append(pos)
        d2d.SetFillPolys(True)
        d2d.SetColour(color)
        d2d.DrawPolygon(ps)
    dos.clearBackground = False
    # d2d.SetFontSize(fontSize)

    #----------------------
    # now draw the molecule, with highlights:
    dict_color_all = {int(idx):[color] for idx in atom_list}
    dict_b_color_all = {int(b):[color] for b in bond_list}
    atomrads_all = {i:atomrads for i in dict_color_all}
    widthmults_all = {i:widthmults for i in dict_b_color_all}
    
    # d2d.DrawMoleculeWithHighlights(mol,"",dict_color_all,
    #                                        dict_b_color_all,
    #                                        atomrads_all, widthmults_all)
    
    d2d.DrawMoleculeWithHighlights(mol,"",dict_color_all,
                                           dict_b_color_all,
                                           atomrads_all, widthmults_all)
    d2d.FinishDrawing()
    pic = d2d.GetDrawingText()
    return pic    

def plot_mol_svg(mol, figsize=(500,500), bondLineWidth=2.5, fontSize=1, addAtomIndex=False, addBondIndex=False, useBWAtomPalette=False):
    """
    Generate an SVG image of a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule to be plotted.
    figsize (tuple, optional): The size of the figure in pixels. Default is (500, 500).
    bondLineWidth (int, optional): The width of the bond lines in pixels. Default is 2.
    fontSize (int, optional): The font size of atom labels in pixels. Default is 16.
    addAtomIndex (bool, optional): Whether to add atom indices to the plot. Default is False.
    addBondIndex (bool, optional): Whether to add bond indices to the plot. Default is False.
    useBWAtomPalette (bool, optional): Whether to use a black and white atom palette. Default is False.

    Returns:
    str: The SVG image of the molecule.
    """
    # Chem.rdDepictor.SetPreferCoordGen(True)
    Chem.rdDepictor.Compute2DCoords(mol)
    drawer = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(figsize[0], figsize[1])

    dos = drawer.drawOptions()
    dos.bondLineWidth = bondLineWidth
    dos.baseFontSize = fontSize
    dos.addAtomIndices = addAtomIndex
    dos.addBondIndices = addBondIndex
    dos.useBWAtomPalette()

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()
    return svg


def get_atom_wise_weight_map(mol, weights, mol_size, cmap='coolwarm'):
    """
    
    Reference:https://github.com/TiagoJanela/MMP-potency-prediction/blob/main/ML/utils_mol_draw.py
    
    usage:
    
    from IPython.display import display, SVG
    pic=get_atom_wise_weight_map(mol, weights, mol_size=(500,500), cmap='coolwarm')
    display(SVG(pic))
    
    
    Generate
    :param mol: molecule to display
    :param weights: weights to map onto the atoms
    :param mol_size: size of the molecule image
    :param cmap: colormap used for weight mapping
    :param return_png: if True: returns PNG image, if False, return Draw.MolDraw2D object
    :return:
    """
    draw2d = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(*mol_size)
    draw2d.drawOptions().fixedScale = 0
    draw2d.drawOptions().useBWAtomPalette()

    mol = Chem.Draw.rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False)
    if not mol.GetNumConformers():
        Chem.rdDepictor.Compute2DCoords(mol)
    if mol.GetNumBonds() > 0:
        bond = mol.GetBondWithIdx(0)
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        sigma = 0.3 * (mol.GetConformer().GetAtomPosition(idx1) -
                       mol.GetConformer().GetAtomPosition(idx2)).Length()
    else:
        sigma = 0.3 * (mol.GetConformer().GetAtomPosition(0) -
                       mol.GetConformer().GetAtomPosition(1)).Length()
    sigma = round(sigma, 2)
    sigmas = [sigma] * mol.GetNumAtoms()
    locs = []
    for i in range(mol.GetNumAtoms()):
        p = mol.GetConformer().GetAtomPosition(i)
        locs.append(Geometry.Point2D(p.x, p.y))
    draw2d.ClearDrawing()
    
    ps = Chem.Draw.ContourParams()
    ps.fillGrid = True
    ps.gridResolution = 0.1
    ps.extraGridPadding = 2.0

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    # it's a matplotlib colormap:
    clrs = [tuple(cmap.get_under()), (1, 1, 1), tuple(cmap.get_over())]
    ps.setColourMap(clrs)

    contourLines = 10
    Chem.Draw.ContourAndDrawGaussians(draw2d, locs, weights, sigmas, nContours=contourLines, params=ps)
    draw2d.drawOptions().clearBackground = False
    draw2d.DrawMolecule(mol)
    draw2d.FinishDrawing()
    
    pic = draw2d.GetDrawingText()
    return pic    


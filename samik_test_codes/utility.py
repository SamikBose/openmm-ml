#!/usr/bin/env python3
"""
Utilities for building a tiny water box, subselecting ~10 waters,
and seeding an excess proton near a chosen oxygen.

NOTE: These helpers are used by the toy all-ANI demo. Not for quantitative use.
"""

import numpy as np
from openmm import unit, Vec3
from openmm.app import Modeller, Topology, ForceField, element

# ---------- helpers ----------
def cube_vectors(L_nm: float):
    """Return cubic box vectors of edge length L_nm (nanometers)."""
    # dont use nanometer here. Newer version of openmm adds nanometer unit by default in the set periodic box command later
    L = L_nm 
    return Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L)

def build_water_box(L_init_nm: float = 1.2) -> Modeller:
    """
    Make a water box (TIP3P) using Amber FF *only to get coordinates*.
    We will not use this FF for forces in the ANI run.
    """
    ff = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    top = Topology()
    modeller = Modeller(top, [])
    L = L_init_nm
    modeller.addSolvent(ff, model='tip3p', boxSize=Vec3(L, L, L))
    return modeller

def list_water_residues(topology):
    names = {"HOH", "WAT", "TIP3", "TIP3P", "SOL"}
    return [res for res in topology.residues() if res.name in names]

def fractional_min_image_delta(dx, L):
    """
    Wrap a 1D displacement dx into [-L/2, L/2] for a cubic box.
    dx, L are unit.Quantity (length).
    """
    Lnm = L
    d = dx
    d -= np.rint(d / Lnm) * Lnm
    return d * unit.nanometer

def keep_n_waters_near_center(modeller: Modeller, N_keep: int = 10, L_final_nm: float = 0.75):
    """
    Keep the N_keep waters closest to the box center; delete the rest.
    Then shrink to a compact cubic box of edge L_final_nm.
    """
    a, b, c = modeller.topology.getPeriodicBoxVectors()
    L = a[0].value_in_unit(unit.nanometer)  # assume cubic
    center = (a + b + c) / 2.0
    pos = modeller.getPositions()

    waters = list_water_residues(modeller.topology)
    ranked = []
    for res in waters:
        O = next(a for a in res.atoms() if a.element == element.oxygen)
        rO = pos[O.index]
        dx = fractional_min_image_delta(rO.x - center[0].value_in_unit(unit.nanometer), L)
        dy = fractional_min_image_delta(rO.y - center[1].value_in_unit(unit.nanometer), L)
        dz = fractional_min_image_delta(rO.z - center[2].value_in_unit(unit.nanometer), L)
        d2 = (dx**2 + dy**2 + dz**2).value_in_unit(unit.nanometer**2)
        ranked.append((d2, res))

    ranked.sort(key=lambda t: t[0])
    keep = set(res for _, res in ranked[:N_keep])
    to_delete_atoms = [atom for res in waters if res not in keep for atom in res.atoms()]
    if to_delete_atoms:
        modeller.delete(to_delete_atoms)

    # set the smaller cubic box
    modeller.topology.setPeriodicBoxVectors(cube_vectors(L_final_nm))

def pick_water_near_center(modeller: Modeller):
    """Return (O_atom, [H1, H2]) for the water closest to box center."""
    a, b, c = modeller.topology.getPeriodicBoxVectors()
    center = (a + b + c) / 2.0
    L = a[0].value_in_unit(unit.nanometer) # cubic
    pos = modeller.getPositions()

    best_O = None
    best_d2 = 1e9
    best_Hs = None

    for res in list_water_residues(modeller.topology):
        atoms = list(res.atoms())
        O = next(a for a in atoms if a.element == element.oxygen)
        Hs = [a for a in atoms if a.element == element.hydrogen]
        rO = pos[O.index]
        dx = fractional_min_image_delta(rO.x - center[0].value_in_unit(unit.nanometer), L)
        dy = fractional_min_image_delta(rO.y - center[1].value_in_unit(unit.nanometer), L)
        dz = fractional_min_image_delta(rO.z - center[2].value_in_unit(unit.nanometer), L)
        d2 = (dx**2 + dy**2 + dz**2).value_in_unit(unit.nanometer**2)
        
        if d2 < best_d2:
            best_O, best_d2, best_Hs = O, d2, Hs

    return best_O, best_Hs

def add_extra_proton_near_oxygen(modeller: Modeller, O_atom, H_atoms, oh_length_nm: float = 0.098):
    """
    Add an extra H ~0.98 Å from O along the bisector of the two water hydrogens.
    No bonds are created (ANI doesn't need bonds); it's just a position.
    """
    pos = modeller.getPositions()
    O = pos[O_atom.index]
    H1 = pos[H_atoms[0].index]
    H2 = pos[H_atoms[1].index]

    # direction along H-H bisector
    v1 = (H1 - O).value_in_unit(unit.nanometer)
    v2 = (H2 - O).value_in_unit(unit.nanometer)
    u = np.array([v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]])
    if np.linalg.norm(u) < 1e-8:
        u = np.random.randn(3)
    u = u / np.linalg.norm(u)

    newH_pos = O + (oh_length_nm * unit.nanometer) * Vec3(u[0], u[1], u[2])

    # add a standalone H atom as a tiny topology and merge into modeller
    tiny = Topology()
    ch = tiny.addChain()
    res = tiny.addResidue("HPLS", ch)  # pseudo-residue for extra proton
    tiny.addAtom("HXP", element.hydrogen, res)
    modeller.add(tiny, [newH_pos])


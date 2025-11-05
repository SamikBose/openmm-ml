import sys
from openmm import unit, LangevinIntegrator
from openmm.app import Simulation, StateDataReporter, PDBReporter
from openmmml import MLPotential
from openmm.app import PDBFile

from utility import (
    build_water_box,
    keep_n_waters_near_center,
    pick_water_near_center,
    add_extra_proton_near_oxygen,
)

import numpy as np
from openmm import unit, Vec3
from openmm.app import Modeller, Topology, ForceField, element

modeller = build_water_box(L_init_nm=1.2)
keep_n_waters_near_center(modeller, N_keep=10, L_final_nm=0.75)
print("Water molecules kept:", sum(1 for r in modeller.topology.residues()
                                   if r.name in {"HOH","WAT","TIP3","TIP3P","SOL"}))

# 2) Build a pure-ANI system (no MM forces).
potential = MLPotential('ani2x')   # or 'ani1ccx'
system = potential.createSystem(modeller.topology)
# Ensure periodic box vectors are set on the System (ANI uses cutoffs under PBC).
a, b, c = modeller.topology.getPeriodicBoxVectors()
if a is not None:
    system.setDefaultPeriodicBoxVectors(a, b, c)
# 3) Integrator & Simulation (tiny timestep, no constraints).
integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.1*unit.femtoseconds)
sim = Simulation(modeller.topology, system, integrator)
sim.context.setPositions(modeller.getPositions())

st0 = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
with open("before_min_no_proton.pdb", "w") as f:
    PDBFile.writeFile(modeller.topology, st0.getPositions(), f)

print("Minimizing...")
sim.minimizeEnergy(maxIterations=500)

st1 = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
with open("after_min_no_proton.pdb", "w") as f:
    PDBFile.writeFile(modeller.topology, st1.getPositions(), f)

# Reporters: stdout every 200 steps, PDB every 500 steps.
sim.reporters.append(StateDataReporter(sys.stdout, 100, step=True, time=True, potentialEnergy=True, temperature=True, speed=True, separator='\t'))
sim.reporters.append(PDBReporter("ani2x_toy_10waters_no_proton.pdb", 500))

print("Running MD...")
sim.step(25000)  # 2.5ps at 0.1 fs
print("Done writing the trajectory...")


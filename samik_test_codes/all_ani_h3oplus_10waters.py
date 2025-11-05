"""
Toy all-ANI MD: ~10 waters + one extra proton (forms H3O+)
================================================================
DEMO ONLY — NOT FOR QUANTITATIVE USE.
- ANI (ANI-1ccx/ANI-2x) lacks explicit long-range electrostatics and pretrained sets target neutral, small molecules.
- An excess proton in water is a charged, condensed-phase system; expect artifacts in energetics/kinetics/structure.
- Use tiny timesteps (≤0.25 fs). If unstable, re-minimize or reduce dt.
"""

import sys
from openmm import unit, LangevinIntegrator
from openmm.app import Simulation, StateDataReporter, PDBReporter
from openmmml import MLPotential

from utility import (
    build_water_box,
    keep_n_waters_near_center,
    pick_water_near_center,
    add_extra_proton_near_oxygen,
)

def main():
    # 1) Build a water box, then carve down to 10 waters and shrink the box.
    modeller = build_water_box(L_init_nm=1.2)
    keep_n_waters_near_center(modeller, N_keep=10, L_final_nm=0.75)
    print("Water molecules kept:", sum(1 for r in modeller.topology.residues()
                                      if r.name in {"HOH","WAT","TIP3","TIP3P","SOL"}))

    # 2) Add an extra proton near the most central water to seed H3O+.
    O_atom, H_atoms = pick_water_near_center(modeller)
    add_extra_proton_near_oxygen(modeller, O_atom, H_atoms)

    # 3) Build a pure-ANI system (no MM forces).
    potential = MLPotential('ani2x')   # or 'ani1ccx'
    system = potential.createSystem(modeller.topology)

    # Ensure periodic box vectors are set on the System (ANI uses cutoffs under PBC).
    a, b, c = modeller.topology.getPeriodicBoxVectors()
    if a is not None:
        system.setDefaultPeriodicBoxVectors(a, b, c)

    # 4) Integrator & Simulation (tiny timestep, no constraints).
    integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.25*unit.femtoseconds)
    sim = Simulation(modeller.topology, system, integrator)
    sim.context.setPositions(modeller.getPositions())

    print("Minimizing...")
    sim.minimizeEnergy(maxIterations=500)

    # Reporters: stdout every 200 steps, PDB every 500 steps.
    sim.reporters.append(StateDataReporter(sys.stdout, 200,
        step=True, time=True, potentialEnergy=True, temperature=True, speed=True, separator='\t'))
    sim.reporters.append(PDBReporter("ani_toy_10waters.pdb", 500))

    print("Running MD...")
    sim.step(5000)  # 1.25 ps at 0.25 fs
    print("Done. Trajectory: ani_toy_10waters.pdb")

if __name__ == "__main__":
    main()

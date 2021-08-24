Multi-state design using AlphaFold2 MCMC hallucination.
====================================================================================
Basile Wicky <basile.wicky@gmail.com>
Started: 2021-08-11
Re-factored and merged with code from Lukas Milles <lmilles@uw.edu>: 2021-08-20 

Summary:
--------
- Designs (hallucinations) are performed by MCMC searches in sequence space and optimizing (user-defined) losses composed of AlphaFold2 metrics and geometric constraints.
- Oligomers with arbitrary number of subunits can be designed.
- Multistate design (either positive or negative) can be specified.
- MCMC trajectories can either be seeded with input sequence(s), or started randomly using AA frequency adjustment.
- At each step, position(s) are chosen for mutation based on different options (see modules/seq_mutation.py for details).
- A 'resfile' (`.af2h` extension) can be employed to specify designable positions and associated probabilities of mutation.
- The oligomeric state (number of subunits) for each oligomer (state) can be specified.
- Specific amino acids can be exluded.
- MCMC paramters (initial temperature, annealing half-life, steps, tolerance) can be specified.
- Currently implemented loss functions are (see modules/losses.py for details):
  - `plddt`: plDDT seem to have trouble converging to complex formation.
  - `ptm`: pTM tends to 'melt' input structures.
  - `pae`: similar to result as ptm?
  - `dual`: combination of plddt and ptm losses with equal weights.
  - `entropy`: current implementation unlikely to work.
  - `pae_sub_mat`: initially made to enforce symmetry, but probably not working.
  - `cyclic`: new trial loss to enforce symmetry based on pae sub-matrix sampling. Not sure it is working. Needs to be benchmarked.
  - `dual_cyclic`: dual with an added geometric loss term to enforce symmetry. Seems to work well.
  - `dual_dssp`: jointly optimises ptm and plddt (equal weights) as well as enforcing a specific secondary structure content as computed by DSSP on the structure.

Minimal inputs:
---------------
- The number and type of subunits for each oligomer, also indicating whether it is a positive or negative design task.
- The length of each protomer or one seed sequence per protomer.

Examples:
-----------
- `./AF2_hallucination_multistate.py --oligo AAAA+,AB+ --L 50,50` will perform 2-state positive design, concomently optimising for a homo-tetramer and a hetero-dimer.
- `./AF2_hallucination_multistate.py --oligo ABC+, --L 40,50,60`  will perform single-state design of a hetero-trimer with protomers of different lengths.
- `./AF2_hallucination_multistate.py --oligo AB+,AA-,BB- --L 50,50` will perform multi-state design concomently optimising for the heterodimer and disfavouring the two homo-dimers.

Outputs:
---------
- PDB structures for each accepted move of the MCMC trajectory.
- A file (.out) containing the scores at each step of the MCMC trajectory (accepted and rejected).

TODO:
------
- A CCE-based loss to enable constrained hallucination based on an input structure?
- RMSD-based or TMscore-based loss to a motif/structure?
- Check AA frequencies.
- Check if normalising pae and pae-derived losses by their init value is an appropriate scaling method?

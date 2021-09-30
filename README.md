# An Optimized Implementation of Quantum Information Set Decoding on Scalable Quantum Resources
## Supplementary code

This repository is related to the paper with the same title in: 

It contains the code necessary to reproduce the circuits specified in the text and run simulations using the Qibo library available in: https://github.com/qiboteam/qibo

The code has been limited to a maximum of 25 qubits in order for them to be run on most classical hardware.

#### This reposotory contains:

- `functions.py`: set of functions and classes used for the creation of the quantum circuits for quantum ISD, amplitude amplification and instance generation.

- `main.py`: solving an instance of a linear code using quantum Information Set Decoding using quantum simultion software.

- `hybrid.sage`: using a hybrid classical-quantum approach to solve instances with limited quantum resources.

#### How to run the code:

`main.py`

- `--n` *(int)* : code length, number of columns of the parity check matrix.
- `--k` *(int)* : code dimension, parity check matrix consists of n-k rows.
- `--sol` *(int)* : number of solutions to force true for the instance.
- `--weight` *(int)* : minimum weight to assert for the instance.
- `--target` *(int)* : target weight for the solution.
- `--num_sol` *(int)* : number of solutions of the instance. Leave blank if not known.
- `--execute` : flag to execute the simulation of the quantum circuit.
- `--iterative` : flag to force the use of the iterative method even when the number of solutions is known.
- `--verbose` : flag to output superposition and oracle sanity checks.
- `--circuit` : flag to output a `.qasm` file with the gates required to solve the quantum ISD instance.
- `--it` *(int)* : number of amplitude amplification iterations for the circuit printed.
- `--summary` : flag to output the number of gates and depth of the quantum circuit.

------------------------------------------------------

`hybrid.sage`

- `--n` *(int)* : code length, number of columns of the parity check matrix.
- `--k` *(int)* : code dimension, parity check matrix consists of n-k rows.
- `--w` *(int)* : error weight of the syndrome decoding instance.
- `--alpha` *(int)* : optimization parameter of hybrid prange, zeros to be guessed.
- `--beta` *(int)* : optimization parameter of punctured hybrid, omitted rows.
- `--p` *(int)* : optimization parameter of punctured hybrid, weight on omitted part.

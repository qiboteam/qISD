import numpy as np
from functions import lin_sys, isd_quantum
from qibo.models import Circuit, Grover
from qibo import gates
import argparse
from scipy.special import binom as binomial


def main(n, k, sol, weight, target, num_sol, iterative, verbose):
    '''Program that solves a linear code using a quantum isd algorithm that makes use of
       quantum amplitude amplification.
    Args:
        n (int): number of columns for the parity check matrix.
        k (int): number of rows of the parity check matrix.
        sol (int): how many solutions to ensure when preparing an instance.
        weight (intr): minimum weight to ensure when preparing an instance.
        target (int): target syndrome weight that solves the problem.
        num_sol (int): number of solutions of the problem (if unknown leave None)
        iterative (bool): force the use of the iterative algorithm.
        verbose (bool): print the checks along the whole process.
        
    Returns:
        solution (list): choice of columns that results in a syndrome with
                         the target weight.
        iterations (int): number of times the oracle has been called in
                          order to reach the solution.
    '''
    instance = lin_sys(n, k, sol, weight)
    if verbose:
        instance.print_instance()

    isd_instance = isd_quantum(instance.H, instance.s)

    if verbose:
        isd_instance.check_superposition()
        isd_instance.check_isd()

    oracle = isd_instance.isd_oracle(target)
    superposition = isd_instance.superposition_circuit()
    sup_size = int(binomial(n, k))
    check = isd_instance.check_solution
    check_args = (isd_instance.H, isd_instance.s, target)

    if num_sol:
        isd_grover = Grover(oracle, superposition_circuit=superposition, superposition_qubits=n,
                            superposition_size=sup_size, number_solutions=num_sol,
                            check=check, check_args=check_args)
    if iterative or not num_sol:
        isd_grover = Grover(oracle, superposition_circuit=superposition, superposition_qubits=n,
                            superposition_size=sup_size, check=check, check_args=check_args, iterative=iterative)

    solution, iterations = isd_grover()
    
    if verbose:
        print(f'Solution found: {solution} in {iterations} Grover iterations.\n')

    return solution, iterations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default=6, type=int)
    parser.add_argument("--k", default=4, type=int)
    parser.add_argument("--sol", default=4, type=int)
    parser.add_argument("--weight", default=2, type=int)
    parser.add_argument("--target", default=3, type=int)
    parser.add_argument("--num_sol", default=None, type=int)
    parser.add_argument("--iterative", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = vars(parser.parse_args())
    main(**args)
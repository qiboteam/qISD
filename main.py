import numpy as np
from functions import lin_sys, isd_quantum
from qibo.models import Grover
import argparse
from scipy.special import binom as binomial


def main(n, k, target, num_sol, execute, iterative, verbose, circuit, it, summary, lee_brickell):
    '''Program that solves a linear code using a quantum isd algorithm that makes use of
       quantum amplitude amplification.
    Args:
        n (int): number of columns for the parity check matrix.
        k (int): number of rows of the parity check matrix.
        target (int): target syndrome weight that solves the problem.
        num_sol (int): number of solutions of the problem (if unknown leave None)
        execute (bool):
        iterative (bool): force the use of the iterative algorithm.
        verbose (bool): print the checks along the whole process.
        circuit (bool): save the gates needed for the circuit in a qasm file for 1 iteration.
        summary (bool): print the summary of the circuit, including depth and gate count.
        it (int): number of iterations to save the gates for.
        lee_brickell (int): number of weight left in the unchecked part. Leave 0 for normal isd.
        
    Returns:
        solution (list): choice of columns that results in a syndrome with
                         the target weight.
        iterations (int): number of times the oracle has been called in
                          order to reach the solution.
    '''
    instance = lin_sys(n, k, lee_brickell=lee_brickell)

    if verbose:
        instance.print_instance()

    isd_instance = isd_quantum(instance.H, instance.s, lee_brickell=lee_brickell)

    if verbose:
        isd_instance.check_superposition()
        isd_instance.check_isd(solutions = instance.solutions)
    if lee_brickell == 0:
        oracle = isd_instance.isd_oracle(target)
    else:
        oracle = isd_instance.lb_oracle(target, lee_brickell)
    superposition = isd_instance.superposition_circuit()
    sup_size = int(binomial(n, k))
    check = isd_instance.check_solution
    check_args = (isd_instance.H, isd_instance.s, target, lee_brickell)

    if num_sol:
        isd_grover = Grover(oracle, superposition_circuit=superposition, superposition_qubits=n,
                            superposition_size=sup_size, number_solutions=num_sol,
                            check=check, check_args=check_args)
    if iterative or not num_sol:
        isd_grover = Grover(oracle, superposition_circuit=superposition, superposition_qubits=n,
                            superposition_size=sup_size, check=check, check_args=check_args, iterative=iterative)
    
    if circuit:
        c = isd_grover.circuit(it)
        gate_list = open(f'quantum_isd_{n}_{k}_lb_{lee_brickell}_gate_list.qasm','w')
        gate_list.write(c.to_qasm())
        print(f'QASM file containing all gates for {it} iterations created.\n')
    
    if summary:
        c = isd_grover.circuit(it)
        print(f'The circuit specifications for {it} iterations are:\n')
        print('-'*30)
        print(c.summary())
        print('-'*30)

    if execute and (isd_grover.nqubits < 29):
        solution, iterations = isd_grover()
        print(f'Solution found: {solution} in {iterations} Grover iterations.\n')
        return solution, iterations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default=6, type=int)
    parser.add_argument("--k", default=4, type=int)
    parser.add_argument("--target", default=2, type=int)
    parser.add_argument("--num_sol", default=None, type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--iterative", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--circuit", action="store_true")
    parser.add_argument("--it", default=1, type=int)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--lee_brickell", default=0, type=int)
    args = vars(parser.parse_args())
    main(**args)
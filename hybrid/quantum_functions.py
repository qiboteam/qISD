from qibo.models import Circuit, Grover
from qibo import gates
import numpy as np
from scipy.special import binom as binomial
import itertools


class isd_quantum:
    '''Constructs quantum primitives needed to solve a linear system using ISD.

    '''

    def __init__(self, H, s):
        '''Sets the specifications for the linear system to solve.
        Args:
            n (int): dimension of the parity check matrix.
            k (int): dimension of the syndrome.
            H (np.array): parity check matrix.
            s (np.array): syndrome.
            
        '''
        def solve_gf2(A, t):
            '''Solve a linear system of equations in the F2 field.
            Args:
                A (np.array): Matrix to solve.
                t (np.array): target vector.
            
            Returns:
                b (np.array): solution of the linear system.
            
            '''
            r = A.shape[0]
            b = np.copy(t)
            for i in range(r):
                for j in range(i+1, r):
                    if A[j, i] == 1:
                        if A[i, i] != 1:
                            A[i] ^= A[j]
                            b[i] ^= b[j]
                        A[j] ^= A[i]
                        b[j] ^= b[i]
                if A[i, i] != 1:
                    raise ValueError("not invertible")
            for i in reversed(range(r)):
                for j in range(i):
                    if A[j, i] == 1:
                        A[j] ^= A[i]
                        b[j] ^= b[i]
            return b
        
        self.H = H
        self.s = s
        self.n = H.shape[1]
        self.k = H.shape[0]
        self.L = self.superposition_probabilities()
        self.n_anc = int(np.ceil(np.log2(self.k+1)))
        self.setup_q_registers()

    def setup_q_registers(self):
        '''Initializes the necessary quantum registers for the instance quantum registers.

        '''
        self.perm = [i for i in range(self.n)]
        self.ancillas = [i for i in range(self.n, self.n+self.n_anc)]
        self.Hq = np.arange(self.n+self.n_anc, self.n+self.n_anc+(self.n-self.k+1)
                           * self.k).reshape(self.n-self.k+1, self.k).transpose()

    def superposition_probabilities(self):
        '''Computes the probabilities to set the initial superposition.
        Returns:
            L (list): position and set values with the corresponding target probability.

        '''
        def split_weights(n, k):
            '''Auxiliary function that gets the required binomials.
            Args:
                n (int): number of sites to fill.
                k (int): number of filled sites.
                
            Returns:
                L (list): position and set values with the corresponding target probability.

            '''
            v0 = binomial(n-1, k)
            v1 = binomial(n-1, k-1)
            return v0/(v0+v1), v1/(v0+v1)

        L = []
        for i in range(self.n):
            for j in range(min(i, self.k-1), -1, -1):
                if self.n-i >= self.k-j:
                    L.append(
                        [self.n-i, self.k-j, split_weights(self.n-i, self.k-j)])
        return L

    def superposition_circuit(self):
        '''Creates an equal quantum superposition over the column choices.
        Returns:
            c (qibo.models.Circuit): quantum circuit with n+n_anc qubits and the gates necessary
                                     to generate the superposition over the column choices.

        '''
        c = Circuit(self.n+self.n_anc)
        c.add(self.set_ancillas_to_num(self.ancillas, self.k))
        tmp = self.n

        for i in self.L:
            if tmp != i[0]:
                c.add(self.sub_one(self.ancillas, [self.n-tmp]))
                tmp = i[0]

            if(i[2] == (0, 1)):
                c.add(self.add_negates_for_check(self.ancillas, i[1]))
                c.add(gates.X(self.n-i[0]).controlled_by(*self.ancillas))
                c.add(self.add_negates_for_check(self.ancillas, i[1]))
            else:
                if i[0] != self.n:
                    c.add(self.add_negates_for_check(self.ancillas, i[1]))
                    c.add(gates.RY(
                        self.n-i[0], float(2*np.arccos(np.sqrt(i[2][0])))).controlled_by(*self.ancillas))
                    c.add(self.add_negates_for_check(self.ancillas, i[1]))
                else:
                    c.add(gates.RY(0, float(2*np.arccos(np.sqrt(i[2][0])))))
        c.add(self.sub_one(self.ancillas, [self.n-1]))
        return c

    def swap_rows(self, i, j, controls):
        '''Swaps needed to handle the first r columns. Swaps the i and j columns depending
           on the controls.
        Args:
            i (int): position of first row.
            j (int): position of second row.
            controls (list): the qubits this operation is controlled by.

        '''
        for col in range(self.n-self.k+1):
            yield gates.SWAP(self.Hq[i, col], self.Hq[j, col]).controlled_by(*controls)

    def set_ancillas_to_num(self, ancillas, num):
        '''Set a quantum register to a specific number.
        Args:
            ancillas (list): ancilliary register to set to a specific number.
            num (int): number to set the ancillas to.

        '''
        ind = 0
        for i in reversed(bin(num)[2:]):
            if int(i) == 1:
                yield gates.X(ancillas[ind])
            ind += 1

    def add_negates_for_check(self, ancillas, num):
        '''Adds the negates needed for control-on-zero for a specific binary number.
        Args:
            ancillas (list): ancillary register that acts as controls.
            num (int): number that needs to be controled.

        '''
        ind = 0
        for i in reversed(bin(num)[2:]):
            if int(i) == 0:
                yield gates.X(ancillas[ind])
            ind += 1
        for i in range(len(bin(num)[2:]), len(ancillas)):
            yield gates.X(ancillas[i])

    def sub_one(self, ancillas, controls):
        '''Subtract 1 bit by bit. Operation controlled by a qubit register.
        Args:
            ancillas (list): quantum register where 1 is substracted.
            controls (list): quantum register that controls the operation.

        '''
        a = ancillas
        yield gates.X(a[0]).controlled_by(*controls)
        for i in range(1, len(a)):
            controls.append(a[i-1])
            yield gates.X(a[i]).controlled_by(*controls)

    def add_one(self, ancillas, controls):
        '''Add one bit by bit. Operation controlled by a qubit register.
        Args:
            ancillas (list): quantum register where 1 is substracted.
            controls (list): quantum register that controls the operation.

        '''
        a = ancillas
        for i in range(len(a)-1):
            controls.append(a[i])
        for i in reversed(range(0, len(a))):
            yield gates.X(a[i]).controlled_by(*controls)
            controls.pop()

    def row_add(self, col, row_res, row_addend, controls):
        '''Matrix row addition in quantum registers. Operation controlled by a qubit register.
        Args:
            col (int): column where the addition is started.
            row_res (int): row where the result is stored.
            row_addend (int): row to be added but not modified.
            controls (list): quantum register that controls the operation.

        '''
        for i in range(col+1, self.n-self.k+1):
            controls.append(self.Hq[row_addend, i])
            yield gates.X(self.Hq[row_res, i]).controlled_by(*controls)
            controls.pop()

    def add_pivot(self, identity, parity, controls):
        '''Set the pivot for the following quantum gaussian elimination algorithm. Operation 
           controlled by a qubit register.
        Args:
            identity (int):
            parity (int):
            controls (list): quantum register that controls the operation.
        
        '''
        i = identity
        p = parity
        yield gates.X(self.Hq[i, p]).controlled_by(*controls)
        # loop for different rows in which the pivot is searched
        for l in range(self.k-(i+1)):
            # current row in which pivot is searched
            pr = i+l+1
            # need to add controls on permutation
            yield self.row_add(p, i, pr, [self.Hq[i, p]]+[self.Hq[i+1+o, p] for o in range(l+1)]+controls)
            # add not for "if all preious were 0"-statement
            if l != self.k-i-2:
                yield gates.X(self.Hq[pr, p]).controlled_by(*controls)
        # revert all nots
        for l in range(self.k-i-1):
            yield gates.X(self.Hq[i+l, p]).controlled_by(*controls)

    def lower_col_elimination(self, identity, parity, controls):
        '''Clear the lower part of the matrix in order to recover the row echelon form. Operation 
           controlled by a qubit register.
        Args:
            identity (int):
            parity (int):
            controls (list): quantum register that controls the operation.
        
        '''
        i = identity
        p = parity
        # for each element below the current diagonal
        for l in range(self.k-i-1):
            # add row if needed to eliminate ones in column p
            yield self.row_add(p, i+l+1, i, [self.Hq[i+l+1, p]]+controls)

    def upper_col_elimination(self, identity, parity, controls):
        '''Used in back substitution in order to find the solution of the system. Operation 
           controlled by a qubit register.
        Args:
            identity (int):
            parity (int):
            controls (list): quantum register that controls the operation.
        
        '''
        i = identity
        p = parity
        # for each row above the current diagonal entry (identiy)
        for l in range(i):
            # eliminate ones by adding row[i] to that row
            yield gates.X(self.Hq[l, self.n-self.k]).controlled_by(*([self.Hq[l, p], self.Hq[i, self.n-self.k]]+controls))

    def initialize_matrix(self):
        '''Initialize the quantum circuit with the chosen matrix.
        Returns:
            c (qibo.models.Circuit): quantum circuit with the gates required to upload the 
                                     parity check matrix to the quantum register.

        '''
        c = Circuit((self.n-self.k+1)*self.k)
        q_reg = np.arange((self.n-self.k+1) *
                          self.k).reshape(self.n-self.k+1, self.k).transpose()
        for i in range(self.k):
            for j in range(self.n-self.k):
                if self.H[i, j+self.k] == 1:
                    c.add(gates.X(q_reg[i, j]))
        for i in range(self.k):
            if self.s[i] == 1:
                c.add(gates.X(q_reg[i, self.n-self.k]))
        return c

    def solve_system(self):
        '''Solve a linear system using a quantum computer.
        Returns:
            c (qibo.models.Circuit): quantum circuit that implements quantum Gaussian elimination
                                     in order to solve a linear system.

        '''
        c = Circuit(self.n+self.n_anc+(self.n-self.k+1)*self.k)
        # add pivots and create upper triangular matrix
        for j in range(self.k-1):
            c.add(self.set_ancillas_to_num(self.ancillas, self.k))
            for i in range(self.n):
                if j < i:
                    c.add(self.add_negates_for_check(self.ancillas, self.k-j))
                    if i < self.k:
                        c.add(self.swap_rows(
                            i, j, self.ancillas+[self.perm[i]]))
                    else:
                        c.add(self.add_pivot(j, i-self.k,
                              self.ancillas+[self.perm[i]]))
                        c.add(self.lower_col_elimination(
                            j, i-self.k, self.ancillas+[self.perm[i]]))
                    c.add(self.add_negates_for_check(self.ancillas, self.k-j))
                c.add(self.sub_one(self.ancillas, [self.perm[i]]))

        # perform backsubstitution
        for j in range(self.k-1):
            c.add(self.set_ancillas_to_num(self.ancillas, self.k))
            for i in reversed(range(self.n)):
                if self.k-j-1 <= i and i >= self.k:
                    c.add(self.add_negates_for_check(self.ancillas, self.k-j))
                    c.add(self.upper_col_elimination(self.k-j-1,
                          i-self.k, self.ancillas+[self.perm[i]]))
                    c.add(self.add_negates_for_check(self.ancillas, self.k-j))
                c.add(self.sub_one(self.ancillas, [self.perm[i]]))
        return c

    def syndrome_weight(self):
        '''Add up the weight of the error syndrome on an ancilla register.
        Returns:
            c (qibo.models.Circuit): quantum circuit that adds up the weight of the syndrome 
                                     into the ancillary register.

        '''
        c = Circuit(self.n+self.n_anc+(self.n-self.k+1)*self.k)
        for i in range(self.k):
            c.add(self.add_one(
                self.ancillas[::-1], [self.Hq[i][self.n-self.k]]))
        return c

    def find_syndrome(self):
        '''Combination of solving the system and adding the syndrome weight.
        Returns:
            c (qibo.models.Circuit): quantum circuit that combines the solution of the linear code
                                     with finding the syndrome weight.

        '''
        c = Circuit(self.n+self.n_anc+(self.n-self.k+1)*self.k)
        c += self.solve_system()
        c += self.syndrome_weight()
        return c

    def isd_oracle(self, target):
        '''Create an oracle that solves quantum isd for a target weight 
           suitable for amplitude amplification.
        Args:
            target (int): target weight of the syndrome that solved the system.
            
        Returns:
            c (qibo.models.Circuit): circuit that implements the oracle for amplitude
                                     amplification. changes the sign of the target weight.

        '''
        c = Circuit(self.n+self.n_anc+(self.n-self.k+1)*self.k+1)
        c.add(self.initialize_matrix().on_qubits(
            *self.Hq.transpose().flatten()))
        c.add(self.find_syndrome().on_qubits(
            *range(self.n+self.n_anc+(self.n-self.k+1)*self.k)))
        c.add(self.add_negates_for_check(self.ancillas[::-1], target))
        c.add(gates.X(c.nqubits-1).controlled_by(*self.ancillas))
        c.add(self.add_negates_for_check(self.ancillas[::-1], target))
        c.add(self.find_syndrome().invert().on_qubits(
            *range(self.n+self.n_anc+(self.n-self.k+1)*self.k)))
        c.add(self.initialize_matrix().invert().on_qubits(
            *self.Hq.transpose().flatten()))
        return c

    def check_solution(self, perm, H, s, target):
        '''Check if a given permutation outputs the desired target weight.
        Args:
            perm (list): choice of columns to check.
            H (np.array): parity check matrix.
            s (np.array): original syndrome.
            target (int): target weight of the syndrome by the end.
            
        Returns:
            (bool): True if the weight of the resulting syndrome is equal to the target.

        '''
        def solve_gf2(A, t):
            '''Solve a linear system of equations in the F2 field.
            Args:
                A (np.array): Matrix to solve.
                t (np.array): target vector.
            
            Returns:
                b (np.array): solution of the linear system.
            
            '''
            r = A.shape[0]
            b = np.copy(t)
            for i in range(r):
                for j in range(i+1, r):
                    if A[j, i] == 1:
                        if A[i, i] != 1:
                            A[i] ^= A[j]
                            b[i] ^= b[j]
                        A[j] ^= A[i]
                        b[j] ^= b[i]
                if A[i, i] != 1:
                    return False
            for i in reversed(range(r)):
                for j in range(i):
                    if A[j, i] == 1:
                        A[j] ^= A[i]
                        b[j] ^= b[i]
            return b

        P = np.matrix(H).transpose()
        L = []
        for i in range(self.n):
            if perm[i] == "1":
                L.append(P[i].tolist()[0])
        Hp = np.matrix(L).transpose()
        x = solve_gf2(Hp, s) % 2
        return np.int(np.sum(x)) == target
    
    
def quantumISD(H, s, target):
    '''Perform a quantum ISD algorithm aided by quantum amplitude amplification.
    Args:
        H (np.array): parity check matrix, prepared for the problem.
        s (np.array): syndrome
        target (int): Hamming weight of the target solution.
        
    Returns:
        solution (str): measured qubits for the solution.
        iterations (int): number of total calls to the oracle until a solution is found.
        
    '''
    qISD = isd_quantum(H, s)
    oracle = qISD.isd_oracle(target)
    superposition = qISD.superposition_circuit()
    sup_size = int(binomial(n, k))
    check = qISD.check_solution
    check_args = (H, s, target)
    isd_grover = Grover(oracle, superposition_circuit=superposition, superposition_qubits=n,
                        superposition_size=sup_size, check=check, check_args=check_args, iterative=iterative)
    solution, iterations = isd_grover()
    return solution, iterations
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
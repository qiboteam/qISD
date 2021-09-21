from qibo.models import Circuit
from qibo import gates
import numpy as np
from scipy.special import binom as binomial
import itertools
from random import shuffle


class lin_sys:
    '''Create an instance of a linear system to solve via ISD.

    '''

    def __init__(self, n, k, lee_brickell=False):
        '''Set the dimension of the system and how many solutions to check for.
        Args:
            n (int): dimension of the parity check matrix.
            k (int): dimension of the syndrome.
            sol (int): number of existing solutions to ensure for the instance.
            weight (int): minimum target weight to ensure.

        '''
        self.n = n
        self.k = k
        self.H, self.s, self.solutions = self.instance_generation(lee_brickell)


    def instance_generation(self, lee_brickell):

        def Reverse(tuples):
            new_tup = tuples[::-1]
            return new_tup

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

        n = self.n
        r = self.k
        solutions=[]
        while(len(solutions)<2):
            solutions=[]
            A=np.random.randint(2, size=(r, n))
            if not(lee_brickell):
                for i in range(r):
                    for j in range(r):
                        if i==j:
                            A[j,n-r+i]=1
                        else:
                            A[j,n-r+i]=0
            while True:
                b=np.random.randint(2, size=(r,1))
                if not(all(b==0)):
                    break
            comb=itertools.combinations([i for i in range(n)],r)
            Aprime=np.arange(r*r).reshape(r,r)
            for i in comb:
                ind=0
                c=0
                for t in i:
                    if t>=n-r:
                        break
                    c+=1
                if not(lee_brickell):
                    new_i=Reverse(i[:c])+i[c:]
                else:
                    new_i=Reverse(i)
                for j in new_i:
                    for l in range(r):
                        Aprime[l,ind]=A[l,j]
                    ind+=1
                mask=0
                for j in i:
                    mask^=(1<<(n-j-1))
                v=bin(mask)[2:]
                while (len(v)<n):
                    v="0"+v
                try:
                    sol=solve_gf2(Aprime,b)
                    for it in reversed(sol):
                        if it==0:
                            v="0"+v
                        else:
                            v="1"+v
                    solutions.append(v)
                except:
                    pass

        return A,b,solutions

    def print_instance(self):
        '''Prints the specifications of the instance.

        '''
        print("Solve all possible linear systems, where\n")
        print("H =")
        print(self.H, '\n')
        print("b =")
        print(self.s, '\n')
        print("Solutions:\n")
        for ss in self.solutions:
            print(
                f'Column choice: {ss[self.s.size:]}     Syndrome: {ss[:self.s.size]}')
        print('\n')


class isd_quantum:
    '''Constructs quantum primitives needed to solve a linear system using ISD.

    '''

    def __init__(self, H, s, lee_brickell=False):
        '''Sets the specifications for the linear system to solve.
        Args:
            n (int): dimension of the parity check matrix.
            k (int): dimension of the syndrome.
            H (np.array): parity check matrix.
            s (np.array): syndrome.

        '''
        self.H = H
        self.s = s
        self.n = H.shape[1]
        self.k = H.shape[0]
        self.L = self.superposition_probabilities()
        self.n_anc = int(np.ceil(np.log2(self.k+1)))
        self.lb = 1
        if lee_brickell:
            self.lb = 0
        self.setup_q_registers()
        self.nq = self.n+self.n_anc + \
            (self.n-self.lb*self.k+1)*self.k + \
            (1-self.lb)*self.n_anc+self.lb*(self.n-self.k)

    def setup_q_registers(self):
        '''Initializes the necessary quantum registers for the instance quantum registers.

        '''
        self.perm = [i for i in range(self.n)]
        self.ancillas = [i for i in range(self.n, self.n+self.n_anc)]
        self.Hq = np.arange(self.n+self.n_anc, self.n+self.n_anc+(self.n-self.lb*self.k+1)
                            * self.k).reshape(self.n-self.lb*self.k+1, self.k).transpose()
        self.counter = [i for i in range(self.n+self.n_anc+(self.n-self.lb*self.k+1)*self.k,
                                         self.n+self.n_anc+(self.n-self.lb*self.k+1)*self.k+(1-self.lb)*self.n_anc)]
        self.aux = [i for i in range(self.n+self.n_anc+(self.n-self.lb*self.k+1)*self.k+(1-self.lb)*self.n_anc,
                                     self.n+self.n_anc+(self.n-self.lb*self.k+1)*self.k+(1-self.lb)*self.n_anc+self.lb*(self.n-self.k))]

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

    def check_superposition(self, nshots=10000):
        '''Checks that the superposition has been created correctly. Prints the choices and 
           their found probability.
        Args:
            nshots (int): samples to take from the probability distribution.

        '''
        c = Circuit(self.n+self.n_anc)
        c += self.superposition_circuit()
        c.add(gates.M(*range(self.n)))
        result = c(nshots=nshots)
        print('-'*37)
        print('| Column choices  | Probability     |')
        print('-'*37)
        for i in result.frequencies():
            print('|', i, ' '*(14-self.n), '|', result.frequencies()
                  [i]/nshots, ' '*(14-len(str(result.frequencies()[i]/nshots))), '|')
            print('-'*37)
        print('\n')

    def swap_rows(self, i, j, controls):
        '''Swaps needed to handle the first r columns. Swaps the i and j columns depending
           on the controls.
        Args:
            i (int): position of first row.
            j (int): position of second row.
            controls (list): the qubits this operation is controlled by.

        '''
        for col in range(self.n-self.lb*self.k+1):
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
        #print('len(a)',len(ancillas))
        #print('num', bin(num)[2:])
        for i in reversed(bin(num)[2:]):
            #print('ind', ind)
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
            ancillas (list): quantum register where 1 is added.
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
        for i in range(col+1, self.n-self.lb*self.k+1):
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
            if self.lb:
                yield gates.X(self.Hq[l, self.n-self.lb*self.k]).controlled_by(*([self.Hq[l, p], self.Hq[i, self.n-self.lb*self.k]]+controls))
            else:
                yield self.row_add(self.k-1, l, i, [self.Hq[l, p]]+controls)

    def initialize_matrix(self):
        '''Initialize the quantum circuit with the chosen matrix.
        Returns:
            c (qibo.models.Circuit): quantum circuit with the gates required to upload the 
                                     parity check matrix to the quantum register.

        '''
        c = Circuit((self.n-self.lb*self.k+1)*self.k)
        q_reg = np.arange((self.n-self.lb*self.k+1) *
                          self.k).reshape(self.n-self.lb*self.k+1, self.k).transpose()
        for i in range(self.k):
            for j in range(self.n-self.lb*self.k):
                if self.H[i, j] == 1:
                    c.add(gates.X(q_reg[i, j]))
        for i in range(self.k):
            if self.s[i] == 1:
                c.add(gates.X(q_reg[i, self.n-self.lb*self.k]))
        return c

    def swap_cols(self, i, j, controls):
        for row in range(self.k):
            yield gates.SWAP(self.Hq[row, i], self.Hq[row, j]).controlled_by(*controls)

    def swap_to_front(self, i, controls):
        for col in reversed(range(i)):
            yield self.swap_cols(col+1, col, controls)

    def swap_bit_to_front(self, i, register, controls):
        for col in reversed(range(i)):
            yield gates.SWAP(register[col+1], register[col]).controlled_by(*controls)

    def find_set_bits(self, x, n):
        bits = []
        for i in range(n):
            mask = 1 << i
            if mask & x:
                bits.append(i)
        return bits

    def col_add(self, col_res, col_addend, controls):
        for i in range(self.k):
            controls.append(self.Hq[i, col_addend])
            yield gates.X(self.Hq[i, col_res]).controlled_by(*controls)
            controls.pop()

    def solve_system_lb(self):
        '''Solve a linear system using a quantum computer.
        Returns:
            c (qibo.models.Circuit): quantum circuit that implements quantum Gaussian elimination
                                     in order to solve a linear system.

        '''
        c = Circuit(self.nq)
        # add pivots and create upper triangular matrix

        for i in self.perm:
            if i != 0:
                c.add(self.swap_to_front(i, [self.perm[i]]))

        for i in range(self.k-1):
            c.add(self.add_pivot(i, i, []))
            c.add(self.lower_col_elimination(i, i, []))
        for i in reversed(range(1, self.k)):
            c.add(self.upper_col_elimination(i, i, []))
        return c

    def solve_system_isd(self):
        c = Circuit(self.nq)

        c.add(self.set_ancillas_to_num(self.ancillas, self.k))

        for j in reversed(range(self.k)):
            for num in reversed(range(j+1, self.k+1)):
                if num != j+1:
                    c.add(self.add_negates_for_check(self.ancillas, num))
                    c.add(self.swap_rows(j, num-1, self.ancillas +
                          [self.perm[j+self.n-self.k]]))
                    c.add(self.add_negates_for_check(self.ancillas, num))
            c.add(self.sub_one(self.ancillas, [self.perm[j+self.n-self.k]]))

        for i in range(self.n-self.k):
            c.add(gates.X(self.aux[i]).controlled_by(self.perm[i]))
            c.add(self.swap_bit_to_front(i, self.aux, [self.perm[i]]))
            c.add(self.swap_to_front(i, [self.perm[i]]))
            c.add(self.sub_one(self.ancillas, [self.perm[i]]))

        for i in range(min(self.k, self.n-self.k)):
            c.add(self.add_pivot(i, i, [self.aux[i]]))
            c.add(self.lower_col_elimination(i, i, [self.aux[i]]))

        for i in reversed(range(1, min(self.k, self.n-self.k))):
            c.add(self.upper_col_elimination(i, i, [self.aux[i]]))

        return c

    def syndrome_weight(self):
        '''Add up the weight of the error syndrome on an ancilla register.
        Returns:
            c (qibo.models.Circuit): quantum circuit that adds up the weight of the syndrome 
                                     into the ancillary register.

        '''
        c = Circuit(self.nq)
        for i in range(self.k):
            c.add(self.add_one(
                self.ancillas[::-1], [self.Hq[i][self.n-self.lb*self.k]]))
        return c

    def find_syndrome(self):
        '''Combination of solving the system and adding the syndrome weight for isd.
        Returns:
            c (qibo.models.Circuit): quantum circuit that combines the solution of the linear code
                                     with finding the syndrome weight.

        '''
        c = Circuit(self.nq)
        c += self.solve_system_isd()
        c += self.syndrome_weight()
        return c

    def enumerate_p_out_of_k(self, weight, p):
        c = Circuit(self.nq)
        r = self.n-self.k
        setb = (1 << p) - 1
        limit = (1 << r)

        while setb < limit:
            columns_to_add = self.find_set_bits(setb, r)
            for i in columns_to_add:
                c.add(self.col_add(self.n, i+self.k, []))

            # CHECK WEIGHTS
            c.add(self.syndrome_weight().on_qubits(*range(c.nqubits)))

            c.add(self.add_negates_for_check(self.ancillas[::-1], weight-p))
            c.add(self.add_one(self.counter[::-1], [*self.ancillas]))
            c.add(self.add_negates_for_check(self.ancillas[::-1], weight-p))

            c.add(self.syndrome_weight().invert().on_qubits(
                *range(c.nqubits)))

            for i in columns_to_add:
                c.add(self.col_add(self.n, i+self.k, []))

            C = setb & - setb
            R = setb + C
            setb = int(((R ^ setb) >> 2) / C) | int(R)

        return c

    def check_isd(self, nshots=10000, solutions=None):
        '''Check that the quantum isd algorithm works as expected.
        Args:
            nshots (int): number of samples to take from the final quantum state.

        '''
        c = Circuit(self.nq)
        c.add(self.superposition_circuit().on_qubits(*range(self.n+self.n_anc)))
        c.add(self.initialize_matrix().on_qubits(
            *self.Hq.transpose().flatten()))
        if self.lb:
            c += self.solve_system_isd()
        else:
            c += self.solve_system_lb()
        c.add(gates.M(*([self.Hq[i, self.n-self.lb*self.k]
              for i in range(self.k)]+self.perm)))
        result = c(nshots=nshots)
        print('-'*55)
        print('| Column choices  | Syndrome        | Probability     |')
        print('-'*55)
        for i in result.frequencies():
            print('|', i[self.k:], ' '*(14-self.n), '|', i[:self.k], ' '*(14-self.k), '|', result.frequencies()[i]/nshots, ' '*(14-len(str(result.frequencies()[i]/nshots))), '|')
            print('-'*55)
        print('\n')
        if solutions:
            if all(i in result.frequencies().keys() for i in solutions):
                print("Circuit works correct")
            else:
                print("Circuit works erroneous")
                a=set([])
                for i in solutions:
                    if i not in m.frequencies().keys() and i not in a:
                        a.add(i)

                print("missing:")
                print(a)

    def isd_oracle(self, target):
        '''Create an oracle that solves quantum isd for a target weight 
           suitable for amplitude amplification.
        Args:
            target (int): target weight of the syndrome that solved the system.

        Returns:
            c (qibo.models.Circuit): circuit that implements the oracle for amplitude
                                     amplification. changes the sign of the target weight.

        '''
        c = Circuit(self.nq+1)
        c.add(self.initialize_matrix().on_qubits(
            *self.Hq.transpose().flatten()))
        c.add(self.find_syndrome().on_qubits(*range(self.nq)))
        c.add(self.add_negates_for_check(self.ancillas[::-1], target))
        c.add(gates.X(c.nqubits-1).controlled_by(*self.ancillas))
        c.add(self.add_negates_for_check(self.ancillas[::-1], target))
        c.add(self.find_syndrome().invert().on_qubits(*range(self.nq)))
        c.add(self.initialize_matrix().invert().on_qubits(
            *self.Hq.transpose().flatten()))
        return c

    def lb_oracle(self, weight, p):
        c = Circuit(self.nq+1)
        c.add(self.initialize_matrix().on_qubits(
            *self.Hq.transpose().flatten()))
        c.add(gates.X(c.nqubits-1))
        c.add(self.enumerate_p_out_of_k(weight, p).on_qubits(*range(self.nq)))
        c.add([gates.X(i) for i in self.counter])
        c.add(gates.X(c.nqubits-1).controlled_by(*self.counter))
        c.add([gates.X(i) for i in self.counter])
        c.add(self.enumerate_p_out_of_k(
            weight, p).invert().on_qubits(*range(self.nq)))
        c.add(self.initialize_matrix().invert().on_qubits(
            *self.Hq.transpose().flatten()))
        return c

    def check_solution(self, perm, A, b, target, lee_brickell=0):
        '''Check if a given permutation outputs the desired target weight.
        Args:
            perm (list): choice of columns to check.
            H (np.array): parity check matrix.
            s (np.array): original syndrome.
            target (int): target weight of the syndrome by the end.

        Returns:
            (bool): True if the weight of the resulting syndrome is equal to the target.

        '''
        def find_set_bits(x, n):
            bits = []
            for i in range(n):
                mask = 1 << i
                if mask & x:
                    bits.append(i)
            return bits

        def solve_gf2(A,t):
            r=A.shape[0]
            b=np.copy(t)
            for i in range(r):    
                for j in range(i+1,r):
                    if A[j,i]==1:
                        if A[i,i]!=1:
                            A[i]^=A[j]
                            b[i]^=b[j]
                        A[j]^=A[i]
                        b[j]^=b[i]
                if A[i,i]!=1:
                    return False
            for i in reversed(range(r)):
                for j in range(i):
                    if A[j,i]==1:
                        A[j]^=A[i]
                        b[j]^=b[i]
            return b

        n = A.shape[1]
        k = A.shape[0]

        P=np.matrix(A).transpose()
        L=[]
        #if lee_brickell:
        #    range_n=[i for i in reversed(range(n))]
        #else:
        #    range_n=[i for i in reversed(range(n-k))]+[i for i in range(k)]
        #for i in range_n:
        #    if perm[i]=="1":
        #        L.append(P[i].tolist()[0])
        for i in range(n):
            if perm[i] == "1":
                L.append(P[i].tolist()[0])
        
        if not(lee_brickell):
            Hp=np.matrix(L).transpose()
            x=solve_gf2(Hp,b)%2
            return np.int(np.sum(x))==target
        else:
            for i in range(n):
                if perm[i]=="0":
                    L.append(P[i].tolist()[0])
            Hp=np.matrix(L).transpose()
            x=solve_gf2(Hp,b)%2
            try:
                len(x)
            except:
                return False
            p=lee_brickell
            correct=False
        
            length=n-k
            setb = (1 << p) - 1;
            limit = (1 << length);
            while setb < limit:
                res=[i for i in x]
                #get positions of set bits in setb
                
                columns_to_add=find_set_bits(setb,k)
                
                #add those columns to syndrome, note that indices need to be shifted by k
                for i in columns_to_add:
                    for j in range(len(res)):
                        res[j]^=Hp[j,k+i]
                if np.int(np.sum(res))==target-p:
                    return True
                    
                #gives next binary number with p bits out of k set to one
                c = setb & - setb;
                v = setb + c;
                setb = int(((v ^ setb) >> 2) / c) | int(v);
            return False
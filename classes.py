from qibo.models import Circuit
from qibo import gates
import numpy as np
from scipy.special import binom as binomial
import itertools


class lin_sys:
    '''Create an instance of a linear system to solve via ISD.
    
    '''
    def __init__(self, n, r, sol):
        '''Set the dimension of the system and how many solutions to check for.
        
        '''
        self.n = n
        self.r = r
        self.sol = sol
        self.A, self.b, self.solutions = self.generate_instance()
        
    
    def generate_instance(self):
        '''Generates specific instance.
        
        '''
        solutions = []
        while(len(solutions)<self.sol):
            solutions=[]
            A=np.random.randint(2, size=(self.r, self.n))
            b=np.random.randint(2, size=(self.r,1))
            comb=itertools.combinations([i for i in range(self.n)],self.r)
            Aprime=np.arange(self.r*self.r).reshape(self.r,self.r)
            for i in comb:
                ind=0
                for j in i:
                    for l in range(self.r):
                        Aprime[l,ind]=A[l,j]
                    ind+=1
                mask=0
                for j in i:
                    mask^=(1<<(self.n-j-1))
                v=bin(mask)[2:]
                while (len(v)<self.n):
                    v="0"+v
                try:
                    solu=np.linalg.solve(Aprime,b)%2
                    for it in reversed(solu):
                        if it==0:
                            v="0"+v
                        else:
                            v="1"+v
                    solutions.append(v)
                except:
                    pass
        return A, b, solutions
    
    
    def print_instance(self):
        '''Prints the specification of the instance.
        
        '''
        print("Solve all possible linear systems, where\n" )
        print("H =")
        print(self.A,'\n')
        print("b =")
        print(self.b,'\n')
        print("Solutions:\n")
        for s in self.solutions:
            print(f'Column choice: {s[self.b.size:]}     Syndrome: {s[:self.b.size]}')
        print('\n')
    
    
class isd_quantum:
    '''Constructs quantum primitives needed to solve a linear system using ISD.
    
    '''
    def __init__(self, n, r, A, b):
        '''
        Sets the specifications of the linear system to solve.
        
        '''
        self.n = n
        self.r = r
        self.A = A
        self.b = b
        self.L = self.superposition_probabilities()
        self.setup_q_registers()
        
        
    def setup_q_registers(self):
        '''Initializes used quantum registers.
        
        '''
        self.perm = [i for i in range(self.n)]
        self.ancillas = [i for i in range(self.n,self.n+self.r)]
        self.H = np.arange(self.n+self.r, self.n+self.r+(self.n+1)*self.r).reshape(self.n+1, self.r).transpose()
        self.ancillas2 = [i for i in range(self.n+self.r+(self.n+1)*self.r, self.n+self.r+(self.n+1)*self.r+int(np.ceil(np.log2(self.r+1))))]        
        
        
    def superposition_probabilities(self):
        '''Computes the probabilities to set the initial superposition.
        
        '''
        def split_weights(n, r):
            '''
            Auxiliary function that gets the required binomials.
            
            '''
            v0=binomial(n-1,r)
            v1=binomial(n-1,r-1)
            return v0/(v0+v1),v1/(v0+v1)
        
        L=[]
        for i in range(self.n):
            for j in range(min(i,self.r-1),-1,-1):
                if self.n-i >= self.r-j:
                    L.append([self.n-i,self.r-j,split_weights(self.n-i,self.r-j)])
        return L
    
    
    def superposition_circuit(self):
        '''Creates an equal quantum superposition over the column choices.
        
        '''
        c = Circuit(self.n+self.r)
        c.add(gates.X(self.n+self.r-1))
        for i in self.L:
            if(i[2]==(0,1)):
                c.add(gates.CNOT(self.n+i[1]-1,self.n-i[0]))
            else:
                if i[0]!=self.n:
                    c.add(gates.CRY(self.n+i[1]-1,self.n-i[0],float(2*np.arccos(np.sqrt(i[2][0])))))
                else:
                    c.add(gates.RY(0,float(2*np.arccos(np.sqrt(i[2][0])))))
            if i[1]!=1:
                c.add(gates.SWAP(self.n-1+i[1],self.n-1+i[1]-1).controlled_by(self.n-i[0]))
            else:
                c.add(gates.CNOT(self.n-i[0],self.n))
        return c
    
    
    def check_superposition(self, nshots=10000):
        '''Checks that the superposition has been created correctly.
        
        '''
        c = Circuit(self.n+self.r)
        c += self.superposition_circuit()
        c.add(gates.M(*range(self.n)))
        result=c(nshots=nshots)
        print('-'*37)
        print('| Column choices  | Probability     |')
        print('-'*37)
        for i in result.frequencies():
            print('|',i,' '*(14-self.n),'|',result.frequencies()[i]/nshots,' '*(14-len(str(result.frequencies()[i]/nshots))),'|')
            print('-'*37)
        print('\n')
    
    
    def set_ancillas_to_num(self, ancillas, num):
        '''Set a quantum register to a specific number.
        
        '''
        ind=0
        for i in reversed(bin(num)[2:]):
            if int(i)==1:
                yield gates.X(ancillas[ind])
            ind+=1
            

    def add_negates_for_check(self, ancillas, num):
        '''Adds the negates needed for control-on-zero.
        
        '''
        ind=0
        for i in reversed(bin(num)[2:]):
            if int(i)==0:
                yield gates.X(ancillas[ind])
            ind+=1
        for i in range(len(bin(num)[2:]),len(ancillas)):
            yield gates.X(ancillas[i])


    def controls_for_check_num(self, ancillas, num):
        '''Specifies control gates to use.
        
        '''
        l=[]
        c=0
        for i in reversed(bin(num)[2:]):
            if int(i)==1:
                l.append(ancillas[c])
            c+=1
        return l

    
    def sub_one(self, ancillas, controls):
        '''Subtract 1 bit by bit.
        
        '''
        a=ancillas
        yield gates.X(a[0]).controlled_by(*controls)
        for i in range(1,len(a)):
            controls.append(a[i-1])
            yield gates.X(a[i]).controlled_by(*controls)


    def add_one(self, ancillas, controls):
        '''Add one bit by bit.
        
        '''
        a=ancillas
        for i in range(len(a)-1):
            controls.append(a[i])
        for i in reversed(range(0,len(a))):
            yield gates.X(a[i]).controlled_by(*controls)
            controls.pop()


    def row_add(self, col, row_res, row_addend, controls):
        '''Matrix row addition in quantum registers.
        
        '''
        for i in range(col+1,self.n+1):
            controls.append(self.H[row_addend,i])
            yield gates.X(self.H[row_res,i]).controlled_by(*controls)
            controls.pop()


    def add_pivot(self, identity, parity, controls):
        i=identity
        p=parity
        yield gates.X(self.H[i,p]).controlled_by(*controls)
        #loop for different rows in which the pivot is searched
        for l in range(self.r-(i+1)):
            #current row in which pivot is searched
            pr=i+l+1
            #need to add controls on permutation
            yield self.row_add(p,i,pr,[self.H[i,p]]+[self.H[i+1+o,p] for o in range(l+1)]+controls)
            #add not for "if all preious were 0"-statement
            if l!=self.r-i-2:
                yield gates.X(self.H[pr,p]).controlled_by(*controls)
        #revert all nots
        for l in range(self.r-i-1):
            yield gates.X(self.H[i+l,p]).controlled_by(*controls)


    def lower_col_elimination(self, identity, parity, controls):
        i=identity
        p=parity
        #for each element below the current diagonal
        for l in range(self.r-i-1):
            #add row if needed to eliminate ones in column p
            yield self.row_add(p,i+l+1,i,[self.H[i+l+1,p]]+controls)


    def upper_col_elimination(self, identity, parity, controls):
        i=identity
        p=parity
        #for each row above the current diagonal entry (identiy)
        for l in range(i):
            #eliminate ones by adding row[i] to that row
            yield gates.X(self.H[l,self.n]).controlled_by(*([self.H[l,p],self.H[i,self.n]]+controls))
            
    
    def initialize_matrix(self):
        '''Initialize the quantum circuit with the chosen matrix.
        
        '''
        c = Circuit((self.n+1)*self.r)
        q_reg = np.arange((self.n+1)*self.r).reshape(self.n+1, self.r).transpose()
        for i in range(self.r):
            for j in range(self.n):
                if self.A[i, j] == 1:
                    c.add(gates.X(q_reg[i, j]))
        for i in range(self.r):
            if self.b[i] == 1:
                c.add(gates.X(q_reg[i, self.n]))
        return c
    
    
    def solve_system(self):
        '''Solve a linear system using a quantum computer.
        
        '''
        c = Circuit(self.n+self.r+(self.n+1)*self.r)
        #add pivots and create upper triangular matrix
        for j in range(self.r-1):
            c.add(self.set_ancillas_to_num(self.ancillas,self.r))
            for i in range(self.n):
                if j<=i:
                    c.add(self.add_negates_for_check(self.ancillas,self.r-j))
                    c.add(self.add_pivot(j,i,self.ancillas+[self.perm[i]]))
                    c.add(self.lower_col_elimination(j,i,self.ancillas+[self.perm[i]]))
                    c.add(self.add_negates_for_check(self.ancillas,self.r-j))
                c.add(self.sub_one(self.ancillas,[self.perm[i]]))
                
        #perform backsubstitution
        for j in range(self.r-1):
            c.add(self.set_ancillas_to_num(self.ancillas,self.r))
            for i in reversed(range(self.n)):
                if self.r-j-1<=i:
                    c.add(self.add_negates_for_check(self.ancillas,self.r-j))
                    c.add(self.upper_col_elimination(self.r-j-1,i,self.ancillas+[self.perm[i]]))
                    c.add(self.add_negates_for_check(self.ancillas,self.r-j))
                c.add(self.sub_one(self.ancillas,[self.perm[i]]))
        return c
    
    
    def syndrome_weight(self):
        '''Calculate the weight of the error syndrome on an ancilla register.
        
        '''
        c = Circuit(self.n+self.r+(self.n+1)*self.r+int(np.ceil(np.log2(self.r+1))))
        for i in range(self.r):
            c.add(self.add_one(self.ancillas2[::-1],[self.H[i][self.n]]))
        return c
    
    
    def find_syndrome(self):
        '''Combination of solving the system and adding the syndrome weight.
        
        '''
        c = Circuit(self.n+self.r+(self.n+1)*self.r+int(np.ceil(np.log2(self.r+1))))
        c.add(self.solve_system().on_qubits(*range(self.n+self.r+(self.n+1)*self.r)))
        c += self.syndrome_weight()
        return c
    
    
    def check_isd(self, nshots=10000):
        '''Check that the isd method for a quantum computer acts as expected.
        
        '''
        c = Circuit(self.n+self.r+(self.n+1)*self.r+int(np.ceil(np.log2(self.r+1))))
        c.add(self.superposition_circuit().on_qubits(*range(self.n+self.r)))
        c.add(self.initialize_matrix().on_qubits(*self.H.transpose().flatten()))
        c += self.find_syndrome()
        c.add(gates.M(*(self.perm+[self.H[i,self.n] for i in range(self.r)]+self.ancillas2)))
        result=c(nshots=nshots)
        print('-'*73)
        print('| Column choices  | Syndrome        | Syndrome weight | Probability     |')
        print('-'*73)
        for i in result.frequencies():
            print('|',i[:self.n],' '*(14-self.n),'|',i[self.n:self.n+self.r],' '*(14-self.r),'|',i[self.n+self.r:],' '*(14-int(np.ceil(np.log2(self.r+1)))),'|',result.frequencies()[i]/nshots,' '*(14-len(str(result.frequencies()[i]/nshots))),'|')
            print('-'*73)
        print('\n')
        
        
    def isd_oracle(self, target):
        '''Create the ISD oracle suitable for Grover's algorithm.
        
        '''
        c = Circuit(self.n+self.r+(self.n+1)*self.r+int(np.ceil(np.log2(self.r+1)))+1)
        c.add(self.initialize_matrix().on_qubits(*self.H.transpose().flatten()))
        c.add(self.find_syndrome().on_qubits(*range(self.n+self.r+(self.n+1)*self.r+int(np.ceil(np.log2(self.r+1))))))
        c.add(self.add_negates_for_check(self.ancillas2,target))
        c.add(gates.X(c.nqubits-1).controlled_by(*self.ancillas2))
        c.add(self.add_negates_for_check(self.ancillas2,target))
        c.add(self.find_syndrome().invert().on_qubits(*range(self.n+self.r+(self.n+1)*self.r+int(np.ceil(np.log2(self.r+1))))))
        c.add(self.initialize_matrix().invert().on_qubits(*self.H.transpose().flatten()))
        return c
    
    
    def check_solution(self, perm, A, b, target):
        '''Check if a given permutation outputs the desired target weight.
        
        '''
        P=np.matrix(A).transpose()
        L=[]
        for i in range(self.n):
            if perm[i]=="1":
                L.append(P[i].tolist()[0])
        Hp=np.matrix(L).transpose()
        x=np.linalg.solve(Hp,b)%2
        return np.int(np.sum(x))==target
        
        
class grover:
    '''Class that performs Grover's algorithm given the necessary parts.
    
    '''
    def __init__(self, superposition, oracle, initial_state=None, sup_qubits=None, sup_size=None, num_sol=None, check=None, check_args=()):
        '''Set the necessary functions and circuits for Grover's algorithm.
        Args:
            superposition (Circuit): quantum circuit that takes an initial state to a superposition. Expected to
                                     use the first set of qubits to store the relevant superposition.
                                     
            oracle (Circuit): quantum circuit that flips the sign using a Grover ancilla initialized with -X-H-
                              and expected to have the total size of the circuit.
            
            initial_state (Circuit): quantum circuit that initializes the state. Leave empty if |000..00>
            
            sup_qubits (int): number of qubits that store the relevant superposition. Leave empty if superpositon
                              does not use ancillas.
                              
            sup_size (int): how many states are in a superposition. Leave empty if its an equal superpositon of quantum states.
            
            num_sol (int): number of expected solutions. Needed for normal Grover. Leave empty for iterative version.
            
            check (function): function that returns True if the solution has been found. Required of iterative approach.
                              First argument should be the bitstring to check.
            
            check_args (tuple): arguments needed for the check function. The found bitstring not included.
            
        '''
        self.superposition = superposition
        self.oracle = oracle
        self.initial_state = initial_state
        if sup_qubits:
            self.sup_qubits = sup_qubits
        else:
            self.sup_qubits = self.superposition.nqubits
        if sup_size:
            self.sup_size = sup_size
        else:
            self.sup_size = int(2**self.superposition.nqubits)
        self.check = check
        self.check_args = check_args
        self.num_sol = num_sol
        self.g_a = self.oracle.nqubits-1
    
    
    def initialize(self):
        '''Initialize the Grover algorithm with the superposition and Grover ancilla.
        
        '''
        c = Circuit(self.oracle.nqubits)
        c.add(gates.X(self.g_a))
        c.add(gates.H(self.g_a))
        c.add(self.superposition.on_qubits(*range(self.superposition.nqubits)))
        return c
        
        
    def diffusion(self):
        '''Construct the diffusion operator out of the superposition circuit.
        
        '''
        c = Circuit(self.superposition.nqubits+1)
        c.add(self.superposition.invert().on_qubits(*range(self.superposition.nqubits)))
        if self.initial_state:
            c.add(self.initial_state.on_qubits(*range(self.initial_state.nqubits)))
        c.add([gates.X(i) for i in range(self.superposition.nqubits)])
        c.add(gates.X(c.nqubits-1).controlled_by(*range(self.superposition.nqubits)))
        c.add([gates.X(i) for i in range(self.superposition.nqubits)])
        if self.initial_state:
            c.add(self.initial_state.invert().on_qubits(*range(self.initial_state.nqubits)))
        c.add(self.superposition.on_qubits(*range(self.superposition.nqubits)))
        return c
    
    
    def step(self):
        '''Combine oracle and diffusion for a grover step.
        
        '''
        c = Circuit(self.oracle.nqubits)
        c += self.oracle
        c.add(self.diffusion().on_qubits(*([*range(self.diffusion().nqubits-1)]+[self.g_a])))
        return c
    
    
    def grover(self, iterations, nshots = 1000):
        '''Perform Grover's algorithm with a set amount of iterations.
        
        '''
        c = Circuit(self.oracle.nqubits)
        c += self.initialize()
        for _ in range(iterations):
            c += self.step()
        c.add(gates.M(*range(self.sup_qubits)))
        result = c(nshots=nshots)
        return result.frequencies(binary=True)
        
        
    def iterative(self):
        '''Iterative approach of Grover for when the number of solutions is not known.
        
        '''
        k = 1
        lamda = 6/5
        self.total_iterations = 0
        while True:
            it = np.random.randint(k+1)
            if it != 0:
                self.total_iterations += it
                result = self.grover(it, nshots=1)
                measured = result.most_common(1)[0][0]
                if self.check(measured, *self.check_args):
                    break
            k = min(lamda*k, np.sqrt(self.sup_size))
            if self.total_iterations > 2*self.sup_size:
                print('Cancelling iterative method as too many iterations have taken place\n')
                break
        return measured
    
        
    def execute(self, freq=False):
        '''Execute Grover's algorithm. If the number of solutions is given, calculates iterations, 
        if not uses an iterative approach.
        
        '''
        if self.num_sol:
            it = np.int(np.pi*np.sqrt(self.sup_size/self.num_sol)/4)
            result = self.grover(it)
            if freq:
                print("Result of sampling Grover's algorihm\n")
                print(result, '\n')
            print(f"Most common states found using Grover's algorithm with {it} iterations:\n")
            most_common = result.most_common(self.num_sol)
            for i in most_common:
                print(i[0], '\n')
                if self.check:
                    if self.check(i[0], *self.check_args):
                        print('Solution checked and successful.\n')
                    else:
                        print('Not a solution of the problem. Something went wrong.\n')
        else:
            if not self.check:
                raise ValueError('Check function needed for iterative approach!\n')
            measured = self.iterative()
            print('Solution found in an iterative process.\n')
            print(f'Solution: {measured}\n')
            print(f'Total Grover iterations taken: {self.total_iterations}\n')
        
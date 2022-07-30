import numpy as np

from qiskit import QuantumCircuit
from qiskit import BasicAer
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector
from numpy.lib.scimath import sqrt as csqrt
from scipy.stats import unitary_group
import itertools

import random
import sys

from qiskit.circuit import QuantumRegister
from qiskit.opflow import StateFn
from qiskit.opflow import Z,X,Y
from qiskit.quantum_info import Pauli
from qiskit.opflow import *
from qiskit.circuit.library import Diagonal
from qiskit.extensions import  UnitaryGate
from qiskit.opflow import I, X, Y, Z
from qiskit.circuit import Parameter
from qiskit.opflow.evolutions import PauliTrotterEvolution,Suzuki
from tqdm import tqdm

def Ising(ground_states, nqbits=8, t=10):
    qubit_idx = list(range(nqbits))
    qubit_pairs = list(itertools.combinations(qubit_idx, 2))
    Js = np.random.normal(0.75, 0.1)
    h_over_Js = 0.1
    h = Js*h_over_Js

    pauli_op = 0
    name_gate=''
    for i in range(nqbits):
        name_gate+= 'I' 
    for pair in qubit_pairs:
        name = name_gate[:pair[0]] + 'Z' + name_gate[(pair[0]+1):pair[1]] + 'Z' + name_gate[(pair[1]+1):]
        if param_choice=='optimal':
            coef = np.random.uniform(-Js/2,Js/2)
        else:
            coef = np.random.normal(0.75, 0.1)
        pauli_op += coef*PauliOp(Pauli(name))

    for qubit in qubit_idx:
        name = name_gate[:qubit] + 'X' + name_gate[(qubit+1):]
        if param_choice=='optimal':
            coef = h
        else:
            coef = np.random.normal(1, 0.1)
        pauli_op += coef*PauliOp(Pauli(name))    

    evo_time = Parameter('θ')
    evolution_op = (evo_time*pauli_op).exp_i()
    trotterized_op = PauliTrotterEvolution(trotter_mode=Suzuki(order=2, reps=1)).convert(evolution_op)
    bound = trotterized_op.bind_parameters({evo_time: t})

    qc_ham = bound.to_circuit()

    obs_res = []
    for i in range(ground_states.shape[0]):

        initial_state = ground_states[i].round(6)
        initial_state/=np.sqrt(np.sum(initial_state**2))

        
        quantum_register = QuantumRegister(nqbits, 'q')

        qc = QuantumCircuit(quantum_register)

        qc.initialize(initial_state, list(range(nqbits)))

        qc.compose(qc_ham, inplace=True)
        
        observables = []
        op_name=''
        for j in range(nqbits):
            op_name+= 'I' 
        for j in range(nqbits):
            # X
            op_nameX = op_name[:j] + 'X' + op_name[(j+1):]
            obs = PauliOp(Pauli(op_nameX))
            observables.append(obs)
            # Y
            op_nameX = op_name[:j] + 'Y' + op_name[(j+1):]
            obs = PauliOp(Pauli(op_nameX))
            observables.append(obs)
            # Z
            op_nameX = op_name[:j] + 'Z' + op_name[(j+1):]
            obs = PauliOp(Pauli(op_nameX))
            observables.append(obs)

        results = []

        backend = BasicAer.get_backend('statevector_simulator')
        job = backend.run(transpile(qc, backend))
        qc_state = job.result().get_statevector(qc)

        for obs in observables:
            obs_mat = obs.to_spmatrix()
            expect = np.inner(np.conjugate(qc_state), obs_mat.dot(qc_state)).real
            results.append(expect)
            
        obs_res.append(results)
    return np.array(obs_res)

if len(sys.argv)!=2:
    raise ValueError('Incorrect number of arguments: ', len(sys.argv))
else:
    name_molecule = str(sys.argv[1])   
    
# Read data
if name_molecule=='LiH':
    with open('training_data/ground_states_LiH.npy', 'rb') as f:
            ground_states = np.load(f)
    nqbits=8
elif name_molecule=='H2O':
    with open('training_data/ground_states_H2O.npy', 'rb') as f:
            ground_states = np.load(f)
    nqbits=10

param_choice='normal'

for j in tqdm(range(100)):
    # Run circuit for all values of ground states:
    obs_res = Ising(ground_states, nqbits)  

    # Store results
    rnd = random.randint(0,9999999)
    filename = str(name_molecule) + '_Ising_' + 'rand'+ str(rnd) + '.npy'
    with open(filename, 'wb') as f:
        np.save(f, obs_res, allow_pickle=True)


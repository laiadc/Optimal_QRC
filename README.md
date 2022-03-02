# Optimal quantum reservoir computing for the NISQ era

In this work,the majorization criterion is used to design the optimal quantum reservoir in terms of performance in QML. The resulting  quantum  circuits  are  easily  realised  in  noisy intermediate-scale  quantum  (NISQ) computers,  and present a significant advantage over the commonly usedIsing model.  The performance of QRC is assessed using different families of quantum circuits, which have different complexity according to the majorization principle.

**Remark**: We recommend the readers to view the notebooks locally and in *Trusted* mode for nicer presentation and correct visualization of the figures. 

The results are illustrated in a set of [Jupyter](https://jupyter.org/) notebooks for easier comprehensiopn. We also provide three python scripts to run the experiments. The whole code is the result of the work in  this paper. Any contribution or idea to continue the lines of the proposed work will be very welcome.

The pipeline used to train the quantum reservoir computing model is the following.  First, the electronic Hamiltonian of the moleculeis mapped to the qubit space,  and its ground state is calculated by direct diagonalization.  Such ground state is fed to the quantum reservoir,  which is a random quantum system sampled from one of the seven families studied in this work.  Local Pauli operators are then measured and fed to the classical machine learning algorithm, which predicts the excited energy of the molecule.  The choice of the quantum reservoir is optimised according to the majorization principle.


<p align="center"><img src="https://github.com/laiadc/Optimal_QRC/blob/main/figures/setting.png"  align=middle width=600pt />
</p>
<p align="center">
<em> Pipeline used to train the quantum reservoir computing model. </em>
</p>

The optimality of the quantum reservoir is illustrated by solving a quantum chemistry problem. In this context, the data used to train the QRC model is already a quantum state. Therefore, it is natural to use a QML algorithm to infer the properties of the system.

## Notebooks

All the notebooks used for this work can be found inside the folder **notebooks** .

### [Generate_data.ipynb](https://github.com/laiadc/Optimal_QRC/blob/main/notebooks/Generate_data.ipynb) 
Generates the training data for the quantum reservoir computing task. The electronic Hamiltonian is mapped to the qubit space and the ground states and excited energies are obtained by direct diagonalization.

### [quantumRC.ipynb](https://github.com/laiadc/Optimal_QRC/blob/main/notebooks/quantumRC.ipynb) 
Illustrates how to design the random quantum circuit for the different families of circuits studied in this work. It also illustrates how to train the quantum machine learning model.

### [quantumRC-Ising.ipynb](https://github.com/laiadc/Optimal_QRC/blob/main/notebooks/quantumRC-Ising.ipynb) 
The performance of the Ising model is assesed. Also, the Ising model is transpiled to a quantum circuit which uses only T, H and CNOT gates. The size of such circuit is studied.




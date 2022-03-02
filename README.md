# Optimal quantum reservoir computing for the NISQ era

In this work,the majorization criterion is used to design the optimal quantum reservoir in terms of performance in QML. The resulting  quantum  circuits  are  easily  realised  in  noisy intermediate-scale  quantum  (NISQ) computers,  and present a significant advantage over the commonly usedIsing model.  The performance of QRC is assessed using different families of quantum circuits, which have different complexity according to the majorization principle.

**Remark**: We recommend the readers to view the notebooks locally and in *Trusted* mode for nicer presentation and correct visualization of the figures. 

The results are illustrated in a set of [Jupyter](https://jupyter.org/) notebooks for easier comprehensiopn. We also provide three python scripts to run the experiments. The whole code is the result of the work in  this paper. Any contribution or idea to continue the lines of the proposed work will be very welcome.

The pipeline used to train the quantum reservoir computing model is the following.  First, the electronic Hamiltonian of the moleculeis mapped to the qubit space,  and its ground state is calculated by direct diagonalization.  Such ground state is fed to the quantum reservoir,  which is a random quantum system sampled from one of the seven families studied in this work.  Local Pauli operators are then measured and fed to the classical machine learning algorithm, which predicts the excited energy of the molecule.  The choice of the quantum reservoir is optimised according to the majorization principle.


<p align="center"><img src="https://github.com/laiadc/Optimal_QRC/blob/main/figures/setting.png"  align=middle />
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

### [Figures.ipynb](https://github.com/laiadc/Optimal_QRC/blob/main/notebooks/Figures.ipynb) 
This notebook analyses the results of the work. 

## Scripts

Additionally, we provide three scripts that can be used to simulate the random quantum circuits used as reservoirs. The scripts `runQC`and `runQC_H2O` can be used to run the simulations for the LiH and H2O molecules respectively. To run them, just type:

`python runQC.py num_gates gates_set observable_type `

where 

+ num_gates is the number of gates (in this work we have used 20, 50, 100, 150, 200)
+ gates_set is the name of the gate set, from the list [G1, G2, G3, MG, D2, D3, Dn]
+ and observable_type must be single

he script `runQC_Ising`and  can be used to run the simulations of the Ising model. To run them, just type:

`python runQC_Ising.py name_molecule `

where 

+ name_molecule is either "LiH" or "H2O"

## Contributions

Contributions are welcome!  For bug reports or requests please [submit an issue](https://github.com/laiadc/PFM_Bearing_Fault_Detection/issues).

## Contact  

Feel free to contact me to discuss any issues, questions or comments.

* GitHub: [laiadc](https://github.com/laiadc)
* Email: [laia.domingo@icmat.es](laia.domingo@icmat.es)

### BibTex reference format for citation for the Code
```
@misc{QRCDomingo,
title={Optimal quantum reservoir computing for the NISQ era},
url={https://github.com/laiadc/Optimal_QRC},
note={GitHub repository containing a criterion of optimality form quantum reservoir computing.},
author={Laia Domingo Colomer},
  year={2022}
}
```


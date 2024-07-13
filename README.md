# Project: Machine learning for Quantum Error Mitigation (QEM)

The goal of this project is to use simple Machine Learning Methods to mitigate Errors occuring in Quantum Computers.
The project is based on the paper "Machine Learning for Practical Quantum Error Mitigation" by IBM (https://arxiv.org/pdf/2309.17368).

To get a first idea of how Quantum Computers work and how to use the package ``qiskit`` refer to the notebooks ``Project_guidelines.ipynb`` and ``VQE_sample_circuit.ipynb``. 

The folders contain different pickle files consisting of one circuit. They were created using the ``random_VQE_data_prep.py`` and ``folded_circuit.py`` for folder ``scaled_pickles`` respectively. In ``extrapolation`` the circuits have layers of the number $6-10$ and otherwise $1-5$. For most of the circuits we used the fake backend FakeQuitoV2 except for the folders containing the name of the backend.

Starting with ``analysis.ipynb`` we only analyzed the original circuits before analyzing in ``analysis_folded.ipynb`` additionally the scaled circuits.
Here we apply all ML models from ``Models.py`` to the data we extract from the circuits using ``FeatureExtract.py`` and plot graphs to show the performance of each model using the $R2$ and $MSE$ as evaluation measurements.

In ``5Models_Test.ipynb`` we analyzed how good a linear Model would fit for every single qubit compared to one general Model.

### Result

We achieved an improvement of $MSE$ compared to the unmitigated error for all models while the Random Forest Regressor seemed to generalize best, also for unseen data.
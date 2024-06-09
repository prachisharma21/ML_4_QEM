from random_VQE_data_prep import *
from mitiq import zne

circ0 = data_loader(file_name="./pickles/circ_0.pickle")
scaled_circuit = zne.scaling.fold_gates_at_random(circ0["Quantum_circuit"][0], 
                                                  scale_factor=1.5,
                                                  fidelities={"single": 1.0, "CNOT": 0.99} )

print("operations in original circuit:", circ0["Quantum_circuit"][0].count_ops())
print("operations in scaled circuit:", scaled_circuit.count_ops())



print("For the original circuit:",circ0["obervable"], circ0["ideal_expectation"],circ0["noisy_expectation"] ) 
obs = circ0["obervable"]
meas_qubit, meas_gate = find_measurement_basis(observable= obs)
meas_qubit 

Noisy_count =Noisy_backend_Simulator(circuits= [scaled_circuit], backend=FakeQuitoV2() , shots = 100000)
noisy_expectation = expectation_value_single_qubit(count = Noisy_count,active_qubit = meas_qubit)
print(f"Noisy expectation value with scaled circuit: {noisy_expectation}")
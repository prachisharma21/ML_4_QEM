from random_VQE_data_prep import *
from mitiq import zne

def fold_circuit(file_path):
    circ = data_loader(file_path)
    scaled_circuit = zne.scaling.fold_gates_at_random(circ["Quantum_circuit"][0], 
                                                    scale_factor=1.5,
                                                    fidelities={"single": 1.0, "CNOT": 0.99} )

    print("operations in original circuit:", circ["Quantum_circuit"][0].count_ops())
    print("operations in scaled circuit:", scaled_circuit.count_ops())



    print("For the original circuit:",circ["obervable"], circ["ideal_expectation"],circ["noisy_expectation"] ) 
    obs = circ["obervable"]
    meas_qubit, meas_gate = find_measurement_basis(observable= obs)
    meas_qubit 

    noisy_count =Noisy_backend_Simulator(circuits= [scaled_circuit], backend=FakeQuitoV2() , shots = 100000)
    noisy_expectation = expectation_value_single_qubit(count = noisy_count,active_qubit = meas_qubit)
    print(f"Noisy expectation value with scaled circuit: {noisy_expectation}")

    return dict({'scaled_circuit': scaled_circuit, 'noisy_count': noisy_count, 'noisy_expectation': noisy_expectation})
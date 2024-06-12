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

def data_storage_Folded_circuit(file_name = 'circ_0.pickle'):
    data_dict = defaultdict() # picking a random noise scaling factor that is above 1 and less than 2 
    noise_scale_factor = round(np.random.uniform(1.01, 1.8, 1)[0],2) 
    single_circ = data_loader(file_name) 
    scaled_circuit = zne.scaling.fold_gates_at_random(single_circ["Quantum_circuit"][0], scale_factor=noise_scale_factor) #fidelities={"single": 1.0, "CNOT": 0.99} ) 
    obs = single_circ["obervable"] 
    meas_qubit, meas_gate = find_measurement_basis(observable= obs) 
    Noisy_count =Noisy_backend_Simulator(circuits= [scaled_circuit], backend=FakeQuitoV2() , shots = 100000) 
    noisy_expectation = expectation_value_single_qubit(count = Noisy_count,active_qubit = meas_qubit) 
    data_dict["Quantum_circuit"] = scaled_circuit 
    data_dict["num_layers"] = single_circ["num_layers"] 
    data_dict["obervable"] = single_circ["obervable"] 
    data_dict["ideal_expectation"] = obs 
    data_dict["noisy_expectation"] = noisy_expectation # extra column which could also be added separately to all the rest of the existing circuits 
    data_dict["scale_factor"] = noise_scale_factor 
    with open(file_name.split(".")[0]+"_scaled.pickle", "wb") as f: 
        pickle.dump(data_dict,f)



data_storage_Folded_circuit()
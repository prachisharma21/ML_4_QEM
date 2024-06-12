# Essential imports for the calculations
# original imports -> changed such that new environment runs
# from qiskit import QuantumCircuit, Aer, transpile
# from qiskit.providers.fake_provider import FakeQuitoV2

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.providers.fake_provider import FakeQuitoV2

from qiskit.quantum_info.operators.symplectic import Pauli
import numpy as np
import pickle

from collections import defaultdict

class CircuitBuilder():
    def __init__(self,params,backend , initial_layout , geometry, nlayers =int):
        self.backend = backend
        self.initial_layout = initial_layout
        self.geometry = geometry        
        self.params = params
        self.num_qubits = len(self.initial_layout)
        self.nlayers = nlayers
       

    def vqeLayer_FakeQuito(self,params): #params has to be a list a list for each layer
        """ VQE layer for the FakeQuito geometry using all qubits and native connectivity"""
        theta_Z = params[0] # make bonds a list of bond list 
        theta_X = params[1]
        theta_ZZ = params[2]
        vqeLayer = QuantumCircuit(self.num_qubits)
        # Choosen bond pairs according to the native qubit connectivity of the backend
        bonds_1 = [[0, 1], [3, 4]]  
        bonds_2 = [[1, 2]]
        bonds_3 = [[1, 3]]
        # the RZ and RZ terms for the field terms of the hamiltonian. 
        # Applied first to get the sequence of layers for PER later to come out correctly, i.e., single qubit gates first followed by clifford gates. 
        
        vqeLayer.rz(theta_Z, range(self.num_qubits))
        vqeLayer.rx(theta_X, range(self.num_qubits))
    
        vqeLayer.cx(*zip(*[bonds_1[i] for i in range(len(bonds_1))]))
        vqeLayer.rz(theta_ZZ, [bonds_1[i][1] for i in range(len(bonds_1))])
        vqeLayer.cx(*zip(*[bonds_1[i] for i in range(len(bonds_1))]))

        vqeLayer.cx(*zip(*[bonds_2[i] for i in range(len(bonds_2))]))
        vqeLayer.rz(theta_ZZ, [bonds_2[i][1] for i in range(len(bonds_2))])
        vqeLayer.cx(*zip(*[bonds_2[i] for i in range(len(bonds_2))]))

        vqeLayer.cx(*zip(*[bonds_3[i] for i in range(len(bonds_3))]))
        vqeLayer.rz(theta_ZZ, [bonds_3[i][1] for i in range(len(bonds_3))])
        vqeLayer.cx(*zip(*[bonds_3[i] for i in range(len(bonds_3))]))

        return vqeLayer
    
    
# Having intial_layout is better---as then you can choose not to use all qubits. 
    def makevqeCircuit_(self, measure = False, meas_basis = "Z"): # NEED to figure out how to input measure and basis here 
        
        vqeCircuit = QuantumCircuit(self.num_qubits)
        vqeCircuit.h(range(self.num_qubits)) # initialize in the |+> state
        vqeCircuit.barrier()
        for i in range(self.nlayers): 
            if self.geometry == "FakeQuitoV2":
                vqeL = self.vqeLayer_FakeQuito(self.params[3*i:3*(i+1)]) 
            elif self.geometry == "FakeGuadalupeV2":
                vqeL = self.vqeLayer_FakeGuadalupeV2(self.params[3*i:3*(i+1)])
            vqeCircuit = vqeCircuit.compose(vqeL)
            vqeCircuit.barrier()
               
        if measure == True:
            if meas_basis == "Z":
                vqeCircuit.measure_all()
                transpiled = transpile(vqeCircuit, self.backend, initial_layout = self.initial_layout)
            elif meas_basis =="X":
                vqeCircuit.h(range(self.num_qubits))
                vqeCircuit.measure_all()
                transpiled = transpile(vqeCircuit, self.backend, initial_layout = self.initial_layout)
            elif meas_basis =='Y':
                vqeCircuit.sdg(range(self.num_qubits))
                vqeCircuit.h(range(self.num_qubits))
                vqeCircuit.measure_all()
                transpiled = transpile(vqeCircuit, self.backend, initial_layout = self.initial_layout)

            else: 
                print("Measurement not defined")    # Y-measurements can be added
        else: 
            transpiled = vqeCircuit # transpile(vqeCircuit, self.backend, initial_layout = self.initial_layout)
        return transpiled
    
    def circ_w_measurements(self, meas_basis = 'Z',meas_qubit = int):
        circ_w_no_meas = self.makevqeCircuit_()
        if meas_basis == "Z":
            circ_w_no_meas.measure_all()
            transpiled = transpile(circ_w_no_meas, self.backend, initial_layout = self.initial_layout)
        elif meas_basis =="X":
            circ_w_no_meas.h(meas_qubit)
            circ_w_no_meas.measure_all()
            transpiled = transpile(circ_w_no_meas, self.backend, initial_layout = self.initial_layout)
        elif meas_basis =='Y':
            circ_w_no_meas.sdg(meas_qubit)
            circ_w_no_meas.h(meas_qubit)
            circ_w_no_meas.measure_all()
            transpiled = transpile(circ_w_no_meas, self.backend, initial_layout = self.initial_layout)

        else: 
                print("Measurement not defined")  

        return transpiled



def State_Vector_Simulator(circuits):
    # The circuits here are without measurements
    count_SV = Aer.get_backend('statevector_simulator').run(circuits[0]).result().get_statevector()
    return count_SV

def QASM_Simulator(circuits,shots=1000):
    count_QASM =Aer.get_backend('qasm_simulator').run(circuits[0], shots=shots).result().get_counts()    
    return count_QASM

def Noisy_backend_Simulator(circuits,backend = FakeQuitoV2(),shots=1000):
    count_Z = backend.run(circuits[0], shots=shots).result().get_counts()
    return count_Z
    

def create_all_single_q_observables(num_qubits = 5,all_pauli = ['X','Y','Z'], num = 1):
    paulis_str = []
    s = "I"*(num_qubits - num) 
    
    for pauli in all_pauli:
        for i in range(num_qubits):
            list_s = list(s)
            list_s.insert(i, pauli)
            paulis_str.append(''.join(list_s))
    return paulis_str

###print(create_all_single_q_observables())

def pick_random_observable():
    # from a list of observables
    observable_list = create_all_single_q_observables()
    observable = np.random.choice(observable_list)
    return observable 


def find_measurement_basis(observable = "IIIIX"):
    basis = list(observable)
    for idx,str in enumerate(basis):
        if str!='I':
            indice = idx
            meas_gate = str
    
    return indice, meas_gate


def expectation_value_single_qubit(count,active_qubit = int):
    count = {tuple(int(k) for k in key):count[key] for key in count.keys()}
    tot = 0
    shots =100000
    for key in count.keys():
            # for ZZ observable
            #num_ZZIII = (-1)**key[4] * (-1)**key[3]
            #num_IZZII = (-1)**key[3] * (-1)**key[2]
            #num_IZIZI = (-1)**key[3] * (-1)**key[1]
            #num_IIIZZ = (-1)**key[1] * (-1)**key[0]

        num = (-1)**key[5-1-active_qubit] 
        tot += num*count[key]
        
    expectation = tot/shots
    return expectation 
    
def Vcircuit_w_random_params(backend =FakeQuitoV2(), n_layers = 1, num_params = 3, meas_basis= "Z" ,meas_qubit= int):

    init_params =  np.random.uniform(-np.pi, np.pi, num_params*n_layers)
    ### print(init_params)
    Vcircuit= CircuitBuilder(params = init_params, backend= backend, initial_layout  = [i for i in range(5)]
                             , geometry="FakeQuitoV2",nlayers =n_layers)
    # Create circuit with no measurements for the state-vector calculations to find the ideal expectation value of the observable
    circ_w_no_meas = [Vcircuit.makevqeCircuit_(measure = False)]
    ### print(circ_w_no_meas[0].draw())
    # Create circuits with specific measurements
    circ_w_meas = [Vcircuit.circ_w_measurements(meas_basis ,meas_qubit)]
    ### print(circ_w_meas[0].draw())

    return circ_w_no_meas, circ_w_meas


def data_preparation():

    # we are choosing Quito fake backend 
    Qbackend = FakeQuitoV2()

    # number of layers can be choosen randomly from 1 to 5
    num_layers = np.random.choice([i+1 for i in range(5)])
    ### print(num_layers)

    # We have only 3 parameters in each layer because of the Mixed field Ising model has 3 parameters. Therefore, it is hard coded
    num_params = 3

    # shots of measurements---reduce this number if it takes longer to run
    num_shots = 100000

    # pick a random observable to measure from a list of all single-qubit measurements
    observable = pick_random_observable()
    ### print("observable= pick_random_observable()", observable)

    # Find the measurement gates and qubit to apply those on. 
    # the following function can be updated later for commuting measurements: For the time it is a simplest implementation
    meas_qubit, meas_gate = find_measurement_basis(observable= observable)

    # create the Quantum circuit with some random parameters 
    # parameters can be extracted from the Q. circuit object later 
    circ_w_no_meas, circ_w_meas = Vcircuit_w_random_params(backend = Qbackend,n_layers = num_layers, num_params = num_params, 
                                                           meas_basis= meas_gate,meas_qubit= meas_qubit)
    
    # performing the Quantum simulations to find the ideal expectation value of the observables
    SV_count = State_Vector_Simulator(circuits=circ_w_no_meas)
    obs = Pauli(observable)
    ideal_expectation = SV_count.expectation_value(obs)
    ### print(ideal_expectation)

    # performing the Quantum simulations to find the noisy expectation value of the observables
    Noisy_count = Noisy_backend_Simulator(circuits=circ_w_meas, backend=Qbackend , shots = num_shots)
    noisy_expectation = expectation_value_single_qubit(count = Noisy_count,active_qubit = meas_qubit)
    ### print(noisy_expectation)

    return [circ_w_meas, num_layers, observable, ideal_expectation, noisy_expectation]



def data_storage(num_circuits_for_ML = 1000):
    # TASK-1
    # run the loop below to create set of 1000 circuits and then spilt that into training and test dataset
    # you could choose if you want to have an evaluation dataset
    data_dict = defaultdict()
    for i in range(num_circuits_for_ML):
        single_circ = data_preparation()
        data_dict["Quantum_circuit"] = single_circ[0]
        data_dict["num_layers"] = single_circ[1]
        data_dict["obervable"] = single_circ[2]
        data_dict["ideal_expectation"] = single_circ[3]
        data_dict["noisy_expectation"] = single_circ[4]
 
        with open(f"circ_{i}.pickle", "wb") as f:
            pickle.dump(data_dict,f)

def data_loader(file_name = str):
    with open(file_name,"rb") as f:
        file = pickle.load(f)
    return file

def main():
    num_circuits_for_ML = 10 # change it to 1000, i.e., circuits for data set. 
    data_storage(num_circuits_for_ML = num_circuits_for_ML)
    loaded_data = data_loader("circ_0.pickle")
    print(loaded_data["Quantum_circuit"][0].__dict__["_data"])
    # from this loaded dictionary object with key Quantum_circuit, one can extract all the other features
    # the rest of the keys are also features for the model
    # TASK 2  
    # Write a function to load each pickle data file and extract the extra features, i.e., count of 2 qubit gates pairs gates as you did in last excercise
    # and create a dataframe with all features included.  
    # One can also use circuits[0].count_ops() to add total gates of each type as features.  


if __name__ == '__main__':
    main()

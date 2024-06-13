import pickle
import pandas as pd
import re
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

def count_gate_paris(df, gates:str):
    l = df[gates]
    res = dict()
    for i in range(len(l)):
        tmp = str(l[i])
        if tmp in res.keys():
            res[tmp] += 1
        else:
            res[tmp] = 1
    return res

def extract_features(path):
    file = open(path, "rb")
    dict_ = pickle.load(file)
    #print(dict)
    #print(len(dict["Quantum_circuit"]))
    data = dict_["Quantum_circuit"][0].data
    #print(dict["Quantum_circuit"][0].draw())
    circuit = dict_["Quantum_circuit"][0]
    

    qubits = str(data)
    #print(qubits)

    # Split input_string before each "CircuitInstruction"
    instructions = qubits.split("CircuitInstruction")

    # Remove the empty string at the beginning (resulting from the initial split)
    instructions = instructions[1:]

    #patterns to get different values from datastring
    pattern_nq = r"num_qubits=(\d+)"
    pattern_nc = r"num_clbits=(\d+)"
    pattern_n = r"name='(\w+)'"
    pattern_p = r"params=\[(.*?)\]"
    pattern_gates = r"Qubit\(QuantumRegister\(5, 'q'\), (\d+)\)"
    #pattern_clb = r'clbits=\(\)\)' -> possibly clbits relevant?

    # Extracted numbers
    # find all values and store them in array to build df
    numbers = [int(match) for match in re.findall(pattern_nq, qubits)]
    clbits = [int(match_cl) for match_cl in re.findall(pattern_nq, qubits)]
    name = [match_name for match_name in re.findall(pattern_n, qubits)]
    params = [match_p for match_p in re.findall(pattern_p, qubits)]

    # loop over all Instructions to get per instruction a list of used qubits
    gates_all = []
    for instr in enumerate(instructions):
        gates = [match_gates for match_gates in re.findall(pattern_gates, str(instr))]
        gates_all.append(gates)


    #create new df to show parameters
    df_new =pd.DataFrame({'name': name, 'num_qubits': numbers, 'num_clbits': clbits, 'params': params, 'gates': gates_all})
    df_new = df_new.sort_values(by='num_qubits', ascending=False)
    df_new = df_new[df_new['num_qubits'] == 2].reset_index()

    res = dict()
    res['noisy_expectation'] = dict_['noisy_expectation']
    res['num_layers'] = dict_['num_layers']

    res['tot_num_gates'] = circuit.size()
    gates = circuit.count_ops()
    res['num_cx_gates'] = gates['cx']
    res['num_sx_gates'] = gates['sx']
    res['num_2_qubit_gates'] = len(df_new)

    

    # encoded obervable as binary number for locations where measured
    input_string = dict_['obervable']
    encoded_string = ''.join(['0' if char == 'I' else '1' for char in input_string])

    # le = LabelEncoder()
    # obs = ran_VQE.create_all_single_q_observables()
    # encoded = le.fit_transform(obs)
    # for index, item in enumerate(obs):
    #     if item == input_string:
    #         res['observable'] = encoded[index]

    res['obs_dummy_1'] = int(not dict_['obervable'][0]=="I")
    res['obs_dummy_2'] = int(not dict_['obervable'][1]=="I")
    res['obs_dummy_3'] = int(not dict_['obervable'][2]=="I")
    res['obs_dummy_4'] = int(not dict_['obervable'][3]=="I")
    res['obs_dummy_5'] = int(not dict_['obervable'][4]=="I")
    
    tmp = count_gate_paris(df_new, 'gates')
    for i in tmp.keys():
        res['count_' + i] = tmp[i]
    res['target'] = dict_['ideal_expectation']

    return res

def extract_features_scaled(path):
    with open(path, "rb") as file:
        dict_ = pickle.load(file)
    #print(dict)
    #print(len(dict["Quantum_circuit"]))
    data = dict_["Quantum_circuit"].data
    #print(dict["Quantum_circuit"][0].draw())
    circuit = dict_["Quantum_circuit"]

    qubits = str(data)
    #print(qubits)

    # Split input_string before each "CircuitInstruction"
    instructions = qubits.split("CircuitInstruction")

    # Remove the empty string at the beginning (resulting from the initial split)
    instructions = instructions[1:]

    #patterns to get different values from datastring
    pattern_nq = r"num_qubits=(\d+)"
    pattern_nc = r"num_clbits=(\d+)"
    pattern_n = r"name='(\w+)'"
    pattern_p = r"params=\[(.*?)\]"
    pattern_gates = r"Qubit\(QuantumRegister\(5, 'q'\), (\d+)\)"
    #pattern_clb = r'clbits=\(\)\)' -> possibly clbits relevant?

    # Extracted numbers
    # find all values and store them in array to build df
    numbers = [int(match) for match in re.findall(pattern_nq, qubits)]
    clbits = [int(match_cl) for match_cl in re.findall(pattern_nq, qubits)]
    name = [match_name for match_name in re.findall(pattern_n, qubits)]
    params = [match_p for match_p in re.findall(pattern_p, qubits)]

    # loop over all Instructions to get per instruction a list of used qubits
    gates_all = []
    for instr in enumerate(instructions):
        gates = [match_gates for match_gates in re.findall(pattern_gates, str(instr))]
        gates_all.append(gates)


    #create new df to show parameters
    df_new =pd.DataFrame({'name': name, 'num_qubits': numbers, 'num_clbits': clbits, 'params': params, 'gates': gates_all})
    df_new = df_new.sort_values(by='num_qubits', ascending=False)
    df_new = df_new[df_new['num_qubits'] == 2].reset_index()

    res = dict()
    res['noisy_expectation'] = dict_['noisy_expectation']
    res['num_layers'] = dict_['num_layers']

    res['tot_num_gates'] = circuit.size()
    gates = circuit.count_ops()
    res['num_cx_gates'] = gates['cx']
    res['num_sx_gates'] = gates['sx']
    res['num_2_qubit_gates'] = len(df_new)

    

    # encoded obervable as binary number for locations where measured
    input_string = dict_['obervable']
    encoded_string = ''.join(['0' if char == 'I' else '1' for char in input_string])

    # le = LabelEncoder()
    # obs = ran_VQE.create_all_single_q_observables()
    # encoded = le.fit_transform(obs)
    # for index, item in enumerate(obs):
    #     if item == input_string:
    #         res['observable'] = encoded[index]

    res['obs_dummy_1'] = int(not dict_['obervable'][0]=="I")
    res['obs_dummy_2'] = int(not dict_['obervable'][1]=="I")
    res['obs_dummy_3'] = int(not dict_['obervable'][2]=="I")
    res['obs_dummy_4'] = int(not dict_['obervable'][3]=="I")
    res['obs_dummy_5'] = int(not dict_['obervable'][4]=="I")
    
    tmp = count_gate_paris(df_new, 'gates')
    for i in tmp.keys():
        res['count_' + i] = tmp[i]
    res['target'] = dict_['ideal_expectation']

    return res

def extract_all_scaled(num_files = 1000):
    a = list()
    for i in range(num_files):
        a.append(extract_features_scaled(f"./scaled_pickles/circ_{i}_scaled.pickle"))
    return pd.DataFrame(a)

def extract_all(num_files=1000):
    a = list()
    for i in range(num_files):
        a.append(extract_features(f"./pickles/circ_{i}.pickle"))
    return pd.DataFrame(a)

def extract_all_extrapolation(num_files=500):
    a = list()
    for i in range(num_files):
        a.append(extract_features(f"./extrapolation/circ_{i}.pickle"))
    return pd.DataFrame(a)

def KF(n_folds = 5, model = None, X = None, y = None):
    kf = KFold(n_splits=n_folds,shuffle=True)
    mse = []
    for i, (train, test) in enumerate(kf.split(X=X)):
        #print('Fold: ' + str(i))
        Xtrain = pd.DataFrame()
        Xtest = pd.DataFrame()
        for key in X.keys():
            Xtrain[key] = X[key][train]
            Xtest[key] = X[key][test]
        ytrain = y.to_numpy()[train]
        ytest = y.to_numpy()[test]
        model = model.fit(Xtrain, ytrain)
        mse.append(mean_squared_error(ytest, model.predict(Xtest)))
    return np.mean(mse)

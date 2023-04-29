import pandas as pd


def build_dataframe(files, headers=None):
    """Creates a pandas dataframe out of multiple data files. Optionally,
    headers can be specified.
    
    Inputs: 
    
    files example:
        ["stats_is_os0_ws0_224_224_256_1.txt", 
        "stats_is_os6_s_ws9_224_224_256_1.txt", 
        "stats_os_ws0_is0_224_224_256_1.txt", 
        "stats_os_ws9_is6_224_224_256_1.txt"]

    headers example (optional): 
        ["simSeconds", 
        "simTicks", 
        "finalTick", 
        "simFreq", 
        "system.cpu.tickCycles"]

    Output: pandas dataframe object
    """
    file_dicts = []

    for file in files:
        file_dicts.append(get_first_simulation_dict(file))

    if headers:
        df = pd.DataFrame(file_dicts, index=files, columns=headers)
    else:
        df = pd.DataFrame(file_dicts, index=files)
    
    return df


def get_simulation_dicts(filename):
    """Returns a list of simulation dictionaries.
    
    Inputs: filename

    Output: [simulation_dict, simulation_dict, ...] where each simulation_dict 
    is a dictionary of a name string mapped to a list of data (also strings):
    {name: [data1], name: [data, data, data]], ...} each "#<description>" is not 
    included in the data
    """
    with open(filename) as f:

        simulation_dicts = []

        raw = f.read()
        lines = raw.split("\n")
        simulations = _get_simulations(lines)
        for simulation in simulations:
            simulation_dicts.append(_make_into_dict(simulation))

        return simulation_dicts
    
def get_first_simulation_dict(filename):
    """Performs the same function as get_simulation_dicts(), but just gets
    the dictionary for the first simulation in the file.
    
    Inputs: filename

    Output: a simulation dict {name: [data1], name: [data, data, data]], ...} 
    each "#<description>" is not included in the data
    """
    with open(filename) as f:

        raw = f.read()
        lines = raw.split("\n")
        simulation =  _get_simulations(lines, first_only=True)
        simulation_dict = _make_into_dict(simulation)
        return simulation_dict


def run_basic_tests(simulation_dicts):
    """Prints the length of the simulation dictionary and the first and last
    data values.

    Inputs: simulation_dicts {name: [data1], name: [data, data, data]], ...}

    Output: None
    """
    for i, simulation_dict in enumerate(simulation_dicts):
        print("\n")
        print("Simulation", i)
        print("lines of data:", len(simulation_dict))
        print("simSeconds:", simulation_dict["simSeconds"])
        print("system.cpu.tickCycles", simulation_dict["system.cpu.tickCycles"])


def _get_simulations(list_of_lines, first_only=False):
    """Takes an unformatted file as a list of lines and returns just the data as 
    a list of simulations
    
    Inputs: list_of_lines [line, line, line, ...]

    Output: simulations [simulation, simulation] where simulations are lists
    of lines [line, line, line, ...]
    """
    simulations = []

    sim_begin = False
    temp = []
    for line in list_of_lines:
        if line == "---------- Begin Simulation Statistics ----------":
            sim_begin = True
        elif line == "---------- End Simulation Statistics   ----------":
            sim_begin = False
            temp.pop()
            simulations.append(temp)
            temp = []
            if first_only:
                return simulations[0]
        elif (sim_begin):
            temp.append(line)
    
    return simulations
# def _get_simulations(list_of_lines, first_only=False):
#     simulations = []
#     temp = []
#     for line in list_of_lines:
#         if line.startswith("simulation"):
#             if temp:  # Add this check
#                 temp.pop()
#                 simulations.append(temp)
#                 temp = []

#             if first_only:
#                 break
#         else:
#             temp.append(line)

#     return simulations

def _make_into_dict(simulation):
    """Takes a simulation and returns it as a dictionary

    Inputs: simulation [line, line, line, ...]

    Output: simulation_dict {name: [data1], name: [data, data, data]], ...}
    """

    simulation_dict = {}


    for line in simulation:
        line_minus_comment = line.split("#")[0]
        name_and_data = line_minus_comment.split()
    
        name = name_and_data[0]
        data = name_and_data[1:]

        if len(data) == 1:
            simulation_dict[name] = float(data[0])
        else:
            simulation_dict[name] = data

    return simulation_dict

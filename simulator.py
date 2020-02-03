from __future__ import print_function
import networkx as nx
import random
from itertools import combinations
import argparse
from scipy.stats import norm
from scipy.stats import uniform

import state_models
from state_models import MasterController
import placement_module


def create_test_topology(host_num, capacity, delay, functions):
    """
    This method creates the host cluster where the controller place the states/replicas into.
     The cluster is modeled as a fully connected graph, where the nodes are the hosts
     and the weights of edges are the delays between the hosts.

    :param host_num:    Required number of the hosts
    :param capacity:    Required memory size of the hosts
    :param delay:       Random variable of the delay among the hosts
    :param functions:   List of functions that are placed into the cluster
    :return:            Topology graph including the hosts and delays and determining which function where to run
    """
    ### Generating topology graph ############################

    G_topology = nx.Graph()

    # Hosts
    for i in range(host_num):
        G_topology.add_node("host_{}".format(i), capacity=capacity, NFs=[], states=[])

    # Edges
    host_pairs = combinations(range(host_num), 2)
    for h1, h2 in host_pairs:
        G_topology.add_edges_from([("host_{}".format(h1), "host_{}".format(h2), {'delay': delay.rvs()})])

    # Mapping functions
    for f in functions:
        host = random.randint(0, host_num - 1)
        G_topology.nodes["host_{}".format(host)]['NFs'].append(f)

    return G_topology


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, action='store', dest='name',
                        help='Name of this test')
    parser.add_argument('-f', '--function_num', type=int, action='store', dest='functions', default=5,
                        help='number of functions')
    parser.add_argument('-s', '--server_num', type=int, action='store', dest='servers', default=3,
                        help='number of servers')
    parser.add_argument('--state_num', type=int, action='store', dest='state_num', default=1,
                        help='number of states to deploy')
    parser.add_argument('--server_capacity', type=int, action='store', dest='server_capacity', default=1000,
                        help='Capacity of the servers')
    parser.add_argument('--max_state_size', type=int, action='store', dest='max_state_size', default=10,
                        help='Maximum size of a state')
    parser.add_argument('--max_slave_num', type=int, action='store', dest='max_slave_num', default=2,
                        help='Maximum number of requested slave replica')
    parser.add_argument('--max_rate', type=int, action='store', dest='max_rate', default=10,
                        help='Maximum rate, i.e. the max number of operation (read or write) in a sec')
    parser.add_argument('--access_pattern_type', type=int, action='store', dest='access_pattern_type', default=1,
                        help='ID of the access pattern')
    parser.add_argument('--with_optimal_calculating', type=bool, dest='opt', default=False,
                        help='Set true if you want calculate the optimal cost as well')
    parser.add_argument('--with_cost_calculating', type=bool, dest='cost_calc', default=False,
                        help='Set false if you want dont want to calculate the costs as well')
    parser.add_argument('--replica_strategy', type=str, dest='rep_strat', default="our",
                        help='Strategy of the replica determination. Possible choices: everywhere, unified, our')
    parser.add_argument('--with_replica_sim', type=bool, dest='rep_sim', default=False,
                        help='Set if you want to compare the replica strats')
    parser.add_argument('--replica_rate', type=int, action='store', dest='replica_rate',
                        help='')
    args = parser.parse_args()

    return args


def set_access_pattern(functions, states, access_type_id, rate_rv):
    """
    This method set the access pattern of the states stored in the 'states' called parameter.
     The access pattern describes who read and/or write the state and how frequent.
    :param functions: list of functions already deployed in the cluster
    :param states:  list of states that are to be placed
    :param access_type_id: ID of the access pattern, i.e., the percentage of readers and/or writers
    :param rate_rv: Random variable of the access frequency
    :return: The summarized number of accesses (read and writes)
    """

    # ID: [Read [%], Write [%], Both [%]]
    access_patter_type = {1: [18, 57, 25],
                          2: [57, 18, 25],
                          3: [33, 33, 33],
                          4: [80, 10, 10]}

    max_function_count = len(functions) - 1

    sum_access_number = 0
    for s in states:

        access_function_count = random.randint(1, max_function_count)
        # Number of reader functions
        reader_count = int(access_function_count * (float(access_patter_type[access_type_id][0]) / 100))
        # Number of writer functions
        writer_count = int(access_function_count * (float(access_patter_type[access_type_id][1]) / 100))
        # Number of functions which both read and write the state 's'
        both_count = int(access_function_count * (float(access_patter_type[access_type_id][2]) / 100))

        # List of Function IDs which both read and write the state 's'
        both_ids = []
        possible_functions = list(range(0, max_function_count))
        while len(both_ids) != both_count:
            id = random.choice(possible_functions)
            both_ids.append(id)
            possible_functions.remove(id)

        # List of Function IDs which read the state 's'
        reader_ids = []
        possible_functions = list(range(0, max_function_count))
        while len(reader_ids) != reader_count:
            id = random.choice(possible_functions)
            if id in both_ids:
                possible_functions.remove(id)
            else:
                reader_ids.append(id)
                possible_functions.remove(id)

        # List of Function IDs which write the state 's'
        writer_ids = []
        possible_functions = list(range(0, max_function_count))
        while len(writer_ids) != writer_count:
            id = random.choice(possible_functions)
            if id in both_ids:
                possible_functions.remove(id)
            elif id in reader_ids:
                possible_functions.remove(id)
            else:
                writer_ids.append(id)
                possible_functions.remove(id)

        # Set states' and functions' reader/writer parameters
        for b in both_ids:
            writer_rate = int(rate_rv.rvs())
            functions[b].add_state_to_write(s, writer_rate)
            reader_rate = int(rate_rv.rvs())
            functions[b].add_state_to_read(s, reader_rate)
            sum_access_number += 1 * writer_rate
            sum_access_number += 1 * reader_rate

        # Set states' and functions' reader parameters
        for r in reader_ids:
            reader_rate = int(rate_rv.rvs())
            functions[r].add_state_to_read(s, reader_rate)
            sum_access_number += 1 * reader_rate

        # Set states' and functions' writer parameters
        for w in writer_ids:
            writer_rate = int(rate_rv.rvs())
            functions[w].add_state_to_write(s, writer_rate)
            sum_access_number += 1 * writer_rate

    return sum_access_number


def create_states(state_count, size, slave_count):
    """
    This method creates the list of states that we'd like to place into our cluster.
    :param state_count: Number of the states to generate
    :param size: Random variable of the size of the states
    :param slave_count: Random variable of slave replica counts
    :return: List of states with random sizes and number of slave replicas. This list is the input of the placement alg.
    """

    states = []
    for s in range(state_count):
        size = int(size.rvs())
        slave_count = int(slave_count.rvs())
        states.append(state_models.MasterState(str(s), size, slave_count))

    return states


# Main function of the placement simulator
if __name__ == "__main__":
    # Parsing arguments
    args = parse_args()
    path = args.name

    print("System parameters:")

    # Creating functions
    functions = [state_models.Function(i, [], []) for i in range(args.functions)]
    print("\t#Function:\t{}".format(args.functions))

    # Creating host cluster
    delay_rv = norm(loc=1, scale=0.1)
    G_topology = create_test_topology(args.servers, args.server_capacity, delay_rv, functions)
    print("\t#Server:\t{}".format(args.servers))
    print("\tCapacity for each server:\t{}".format(args.server_capacity))
    print("\tMean of Delays:\t\t{}".format(delay_rv.mean()))

    # Creating states
    state_size_rv = uniform(loc=1, scale=args.max_state_size)
    slave_num_rv = uniform(loc=1, scale=args.max_slave_num)
    states = create_states(args.state_num, state_size_rv, slave_num_rv)
    print("\t#State:\t\t{}".format(args.state_num))
    print("\tMax state size:\t{}".format(args.max_state_size))
    print("\tMax replica#:\t{}".format(args.max_slave_num))

    # Setting access pattern
    rate_rv = uniform(loc=1, scale=args.max_rate - 1)
    sum_access_number = set_access_pattern(functions, states, args.access_pattern_type, rate_rv)
    print("SUM ACCESS COUNT: {}".format(sum_access_number))
    print("\tAccess pattern:\t{}".format(args.access_pattern_type))
    print("\tMax rate:\t{}".format(args.max_rate))

    # Creating request including states to place
    req = state_models.Request(states)

    # Declaring Placement controller node
    master_controller_node = MasterController(G_topology, functions, placement_module.PlacementModule(), delay_rv)

    # Deploying request
    master_controller_node.events.append(
        state_models.Event(state_models.EventTypes.PLACE_STATES, req, None, None, None))

    costs, runtimes = master_controller_node.run(args.opt)

    print("END OF SIMULATING")

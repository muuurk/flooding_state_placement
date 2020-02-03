
from __future__ import print_function
import state_models
import bellmanford as bf
import datetime
from state_models import MasterController
import os
import docplex.mp.model as cpx
import subprocess
import random

CPLEX_PATH = "/home/xxx/projects/cplex/cplex/bin/x86-64_linux/cplex"
CPLEX_TIME_LIMIT = 600


class NodeAsPlacementDest:

    def __init__(self, node_id, capacity, virtual_elements):
        """
        Constructor of NodeAsPlacementDest
        :param node_id:             ID of the host
        :param load:                The load of the host (e.g. used RAM)
        :param capacity:            Capacity of the host (e.g. capacity of the RAM)
        :param virtual_elements:    List of elements deployed to the host
        """
        self.node = node_id
        self.capacity = capacity

        sum_load = 0
        for s in MasterController.topology.nodes[node_id]['states']:
            sum_load += s.size
        self.load = sum_load
        self.virtual_elements = virtual_elements


class PlacementModule:
    current_req = None
    delay_matrix = None
    mapping = None
    reading_access_rates = dict()
    writing_access_rates = dict()

    def get_node_of_state(self, state):
        """
        Returns the id of the host where the state is located.
        :param state: state to which we are looking for the host where it is stored
        :return: host id
        """
        try:
            for node, values in self.mapping.items():
                if state in values.virtual_elements:
                    return node
        except StopIteration as e:
            print("Function {} is not deployed in the given topology".format(function))
            raise e

    def get_most_loaded_host(self):
        """
        This method returns the host with the minimum free capacity.
        :return: host id
        """
        free_space = float('inf')
        most_loaded_node = None
        for node, values in self.mapping.items():
            if free_space > (values.capacity - values.load):
                free_space = (values.capacity - values.load)
                most_loaded_node = node
        return most_loaded_node

    def get_state_to_move(self, host, banned_states):
        """
        This method return the state what the flooding algorithm migrate to a new host. This migration is neccessary, as
        the host is overloaded, i.e., its free capacity is less than zero.
        :param host: the overloaded host id
        :param banned_states: list of states what we cannot choose to migrate
        :return: state to migrate
        """

        if self.mapping[host].capacity - self.mapping[host].load >= 0:
            return None

        states = []
        replicas = []

        for s in self.mapping[host].virtual_elements:

            moving_cost = 0

            if isinstance(s, state_models.MasterState):
                access_functions = s.readers + s.writers
                # access_functions.extend(s.writers)
                access_functions = list(dict.fromkeys(access_functions))
                for f in access_functions:
                    moving_cost += self.get_function_rate(s, f)
                states.append({'state': s, 'cost': moving_cost})

            if isinstance(s, state_models.SlaveReplica):
                access_functions = s.master.readers + s.master.writers
                # access_functions.extend(s.master.writers)
                access_functions = list(dict.fromkeys(access_functions))
                for f in access_functions:
                    moving_cost += self.get_function_rate(s.master, f)
                replicas.append({'state': s, 'cost': moving_cost})

        # sort replicas according to their moving cost
        replicas = sorted(replicas, key=lambda i: i['cost'])
        for r in replicas:
            if r['state'].id not in banned_states:
                return r['state']

        # sort states according to their moving cost
        states = sorted(states, key=lambda i: i['cost'])
        for s in states:
            if s['state'].id not in banned_states:
                return s['state']

        # FIXME:
        return "error"

    def generating_delay_matrix(self):
        """
        This method generated a delay matrix of the cluster
        :return: A dictionary as delay matrix
        """
        d = {(i, j): float('inf') for i in self.topology.nodes for j in self.topology.nodes}
        for i in self.topology.nodes:
            for j in self.topology.nodes:
                path_length, path_nodes, negative_cycle = bf.bellman_ford(self.topology, source=i, target=j,
                                                                          weight="delay")
                d[(i, j)] = path_length
        return d

    def get_function_rate(self, state, f, access_type='both'):
        """
        This method returns the access rate of function 'f' which accesses data 'state'
        :param state: The state that read and/or write the function 'f'
        :param f: The function that access the state
        :return: The access rate i.e., the access count of state per sec initited by function 'f'
        """
        if isinstance(state, state_models.MasterState):
            master = state
        elif isinstance(state, state_models.SlaveReplica):
            master = state.master

        try:
            reading_rate = self.reading_access_rates[(f.id, master.id)]
        except KeyError:
            for d in f.readings:
                self.reading_access_rates.update({(f.id, d['state'].id): d['rate']})
            try:
                reading_rate = self.reading_access_rates[(f.id, master.id)]
            except KeyError:
                self.reading_access_rates.update({(f.id, master.id): 0})
                reading_rate = self.reading_access_rates[(f.id, master.id)]

        try:
            writing_rate = self.writing_access_rates[(f.id, master.id)]
        except KeyError:
            for d in f.writings:
                self.writing_access_rates.update({(f.id, d['state'].id): d['rate']})
            try:
                writing_rate = self.writing_access_rates[(f.id, master.id)]
            except KeyError:
                self.writing_access_rates.update({(f.id, master.id): 0})
                writing_rate = self.writing_access_rates[(f.id, master.id)]

        if access_type == 'both':
            return reading_rate + writing_rate
        elif access_type == 'reading':
            return reading_rate
        elif access_type == 'writing':
            return writing_rate
        else:
            raise ("Access type {} is unknown.".format(access_type))

    def ordering_states(self, states):
        """
        Sorting states according to their total size
        :param states: list of master states
        :return: ordered list of master states
        """

        tmp_list = []
        for s in states:
            functions_of_state = [i for i in s.readers] + [i for i in s.writers]
            functions_of_state = list(dict.fromkeys(functions_of_state))

            sum_size = 0
            for f in functions_of_state:
                sum_size += self.get_function_rate(s, f)
            tmp_list.append({'obj': s, 'size': sum_size})

        tmp_list = sorted(tmp_list, key=lambda i: i['size'], reverse=True)
        return [state['obj'] for state in tmp_list]

    def init_tmp_mapping(self, masters):
        """
        Creating the list of Host objects where we can temporary store the states during the placement
        :param hosts: list of hosts
        :return: sorted list (by capacity) of hosts
        """

        tmp_host_list = []
        for h in list(self.topology.nodes):
            tmp_host_list.append(NodeAsPlacementDest(h, self.topology.nodes[h]["capacity"],
                                                     self.topology.nodes[h]["states"] + self.topology.nodes[h]["NFs"]))
        tmp_host_list = sorted(tmp_host_list, key=lambda i: i.capacity, reverse=True)

        hosts_where_to_map = {}
        for i in tmp_host_list:
            hosts_where_to_map.update({i.node: i})

        self.mapping = hosts_where_to_map
        for m in masters:
            for h, details in self.mapping.items():
                for ve in details.virtual_elements:
                    if isinstance(ve, state_models.MasterState):
                        if ve.id == m.id:
                            self.mapping = self.delete_state_from_mapping(m, h)
                    elif isinstance(ve, state_models.SlaveReplica):
                        if ve.master.id == m.id:
                            self.mapping = self.delete_state_from_mapping(ve, h)

        return self.mapping

    def delete_state_from_mapping(self, state, host):
        """
        Delete a given state from the temporary placement
        :param state: state to be deleted
        :param host: host which contains the state
        :return: the new placement
        """

        state_to_delete = next(i for i in self.mapping[host].virtual_elements if i.id == state.id)
        self.mapping[host].virtual_elements.remove(state_to_delete)
        self.mapping[host].load = self.mapping[host].load - state.size

        return self.mapping

    def getOptimalHosts(self, hosts, hosts_of_functions, state, banned_hosts=[], capacity_check=False,
                        access_type='both'):
        """
        It returns the optimal  host  is  found  by  a  shortest  path  search (Alg. 2 in the paper)
        e.g. using the Floyd Warshall algorithm
        :param hosts:                   set of candidate hosts where the state can be placed
        :param hosts_of_functions:      set of hosts from where the NFs access the state
        :return:                        optimal host and its cost to the state be mapped there
        """

        min_cost = float("inf")
        min_host = None
        closest_nodes = []

        hosts = [h for h in hosts if 'host' in h]

        for candidate_host in hosts:
            if candidate_host not in banned_hosts:
                if capacity_check:
                    if (self.topology.nodes[candidate_host]['capacity'] - self.mapping[
                        candidate_host].load) < state.size:
                        continue
                cost = 0
                for h in hosts_of_functions:
                    functions = self.topology.nodes[h]['NFs']
                    rate = 0
                    for f in functions:
                        rate += self.get_function_rate(state, f, access_type=access_type)
                    cost += (self.delay_matrix[(candidate_host, h)]) * rate
                closest_nodes.append({'node': candidate_host, 'cost': cost})
                if cost < min_cost:
                    min_cost = cost
                    min_host = candidate_host

        if min_host == None:
            raise Exception('Too much replica number request. '
                            'There is no enough host where the all the replicas could be deployed.')

        closest_nodes = sorted(closest_nodes, key=lambda i: i['cost'])

        return min_host, min_cost, closest_nodes

    def get_free_capacity(self, host):
        """
        Returns the free capacity of the given host
        :param host:
        :return:
        """
        return host.capacity - host.load

    def print_current_mapping(self):
        """
        Print the current status of the placement, i.e., the list of hosts and the contained state replicas
        :return: None
        """
        # for node, values in self.mapping.items():
        #     virtual_elements = []
        #     for i in self.mapping[node].virtual_elements:
        #         try:
        #             if 'function' not in i.id:
        #                 virtual_elements.append(i.id)
        #         except:
        #             if 'function' not in i:
        #                 virtual_elements.append(i)
        #     print("{}: {} \t| {}".format(node, values.capacity - values.load, virtual_elements))
        # print("-----------------------------------------------------------------")
        return

    def get_access_functions(self, state, access_type='both'):
        """
        Returns the access functions of the given state.
        :param state:
        :param access_type: filter the access functions according to the access type (read, write or both)
        :return: list of access functions
        """
        if isinstance(state, state_models.MasterState):
            if access_type == 'both':
                function_set = state.readers + state.writers
            elif access_type == 'read':
                function_set = state.readers
            else:
                function_set = state.writers
            function_set = list(dict.fromkeys(function_set))
            return function_set

        if isinstance(state, state_models.SlaveReplica):
            if access_type == 'both':
                function_set = state.master.readers + state.master.writers
            elif access_type == 'read':
                function_set = state.master.readers
            else:
                function_set = state.master.writers
            function_set = list(dict.fromkeys(function_set))
            return function_set

    def move_is_okay(self, src_host, dst_host, state):
        """
        Returns wether migrating a state is possible or not
        :param src_host: the host from where the state could be moved
        :param dst_host: the host where the state could move to
        :param state: state to be moved
        :return: True or False
        """
        if self.get_free_capacity(src_host) < 0:
            if self.get_free_capacity(dst_host) - state.size >= 0:
                if not self.is_aa_node(state, dst_host):
                    return True
        else:
            if self.get_free_capacity(src_host) < self.get_free_capacity(
                    dst_host) - state.size:
                if not self.is_aa_node(state, dst_host):
                    return True
        return False

    def is_aa_node(self, state, host):
        """
        Check whether the given host is appropriate to the state to store it, i.e., it does not contain slaves of the given state.
        :param state: state to be moved
        :param host: host where the state could be moved to
        :return: True or False
        """
        if isinstance(state, state_models.MasterState):
            replicas = self.current_req.get_replicas_of_state(state)
            for r in replicas:
                if r in host.virtual_elements:
                    return True

        elif isinstance(state, state_models.SlaveReplica):
            if state.master in host.virtual_elements:
                return True
            replica_mates = self.current_req.get_replicas_of_state(state.master)
            for r in replica_mates:
                if r in host.virtual_elements:
                    return True
        return False

    def deploy_states(self):
        """
        Place the states according to the temporary placement to the real topology model.
        :return: The real topology containing the requested states.
        """
        # host: ID of the host, host_data: All info related to the host
        for host, host_data in self.mapping.items():
            # ve: virtual elements deployed to the host (functions, states, replicas)
            for ve in host_data.virtual_elements:
                if not isinstance(ve, state_models.Function):
                    if ve.id not in [i.id for i in MasterController.topology.nodes[host]['states']]:
                        MasterController.topology.nodes[host]['states'].append(ve)
            for s in MasterController.topology.nodes[host]['states']:
                if s.id not in [i.id for i in host_data.virtual_elements]:
                    MasterController.topology.nodes[host]['states'].remove(s)

    def get_host_of_function(self, function):
        """
        Returns the host where the given function is located.
        :param function:
        :return:
        """

        if isinstance(function, state_models.Function):
            function_id = function.id
        else:
            function_id = function

        try:
            node = next(h for h in self.topology.nodes if function_id in [i.id for i in self.topology.nodes[h]['NFs']])
        except StopIteration as e:
            print("Function {} is not deployed in the given topology".format(function_id))
            raise e
        return node

    def check_sum_capacity(self, request):
        """
        Checks whether the size of the requested states is greater than the free capacity of the cluster
        :param request:
        :return:
        """
        sum_capacity = 0
        for h in self.mapping:
            sum_capacity += self.mapping[h].capacity
        sum_req_size = 0
        for r in request.states:
            sum_req_size += r.size
        print("SUM CAPACITY: {} \t SUM_REQ_SIZE: {}".format(sum_capacity, sum_req_size))
        if sum_req_size > sum_capacity:
            raise Exception("Sum capacity is less then the sum size of the requests")

    def place_request_flooding(self, req):

        t1 = datetime.datetime.now()
        print("### FLOODING Placement module #################################################\n")
        # Init variables describing the current state of the infrastructure
        self.topology = MasterController.topology
        self.delay_matrix = self.generating_delay_matrix()

        # Init variables
        self.current_req = req
        masters = req.get_masters()
        hosts = self.topology.nodes
        hosts = [h for h in hosts if 'host' in h]
        self.mapping = self.init_tmp_mapping(masters)

        # if sum capacity is less than the sum requests size raise an Exception
        self.check_sum_capacity(req)

        ###### Orchestration ###############################################################################################

        self.print_current_mapping()

        # Ordering states - Line 3 from pseudocode
        sorted_masters = self.ordering_states(masters)

        all_assignments = []
        # Creating temporary mappings without checking hosts' capacity
        for s in sorted_masters:

            # replica set of (master) state 's'                                                       || pseudocode:5
            replica_set = req.get_replicas_of_state(s)

            # set of functions accessing state 's'. Access states are the readers and also  the writers || pseudocode:6
            function_set = self.get_access_functions(s)

            # set of hosts where the accessing functions are located || pseudocode:7
            host_set = [h for h in hosts for f in function_set if f in self.topology.nodes[h]['NFs']]

            # if optimal placement solution can be found in polinomial time

            # if there is no replica || pseudocode:8
            if s.replica_num == 0:
                candidate_host, cost, a = self.getOptimalHosts(hosts, host_set, s)
                self.mapping[candidate_host].load += s.size
                self.mapping[candidate_host].virtual_elements.append(s)

            # If the state does have slave replica(s), and is read by only one function instance      || pseudocode:11
            elif len(s.readers) == 1:
                candidate_host_for_master, cost, a = self.getOptimalHosts(hosts, host_set, s)  # || pseudocode:12
                self.mapping[candidate_host_for_master].load += s.size
                self.mapping[candidate_host_for_master].virtual_elements.append(s)
                banned_hosts = [candidate_host_for_master]
                for r in replica_set:  # || pseudocode:14
                    candidate_host_for_slave, cost, a = self.getOptimalHosts(hosts, [candidate_host_for_master], s,
                                                                             banned_hosts)
                    self.mapping[candidate_host_for_slave].load += r.size
                    self.mapping[candidate_host_for_slave].virtual_elements.append(r)
                    banned_hosts.append(candidate_host_for_slave)

            # If the state does have slave replica(s), and is read by multiple function instance
            else:

                reader_functions = s.readers  # || pseudocode:18
                tmp_list = [{'obj': i, 'rate': j['rate']} for i in reader_functions for j in i.readings if
                            j['state'] == s]
                reader_functions = sorted(tmp_list, key=lambda i: i['rate'], reverse=True)  # || pseudocode:19
                reader_functions = [i['obj'] for i in reader_functions]
                # Writers
                writer_functions = s.writers  # || pseudocode:20

                # list of state/replica <-> reader functions assignments
                assignments = []

                # If more RO functions exist than the required replica number
                # Spread out replicas over the functions
                if s.replica_num < len(reader_functions):  # || pseudocode:23
                    for i in range(s.replica_num):
                        assignments.append((replica_set[i], reader_functions[i]))
                    for i in range(s.replica_num + 1, len(reader_functions)):
                        assignments.append((s, reader_functions[i]))

                # If the replica number is greater then the number of RO functions
                # Spread out replicas over the functions
                elif s.replica_num >= len(reader_functions):  # || pseudocode:28
                    for i in range(len(reader_functions)):
                        assignments.append((replica_set[i], reader_functions[i]))

                # Assign other (write-only) functions
                for f in writer_functions:  # || pseudocode:32
                    assignments.append((s, f))

                # Assign master state to slaves
                for i in range(len(replica_set)):  # || pseudocode:34
                    assignments.append((replica_set[i], s))

                # banned hosts list will contain the host, where the master state and the other replicas exist
                banned_hosts = []
                list_of_state_n_replica = [s]
                list_of_state_n_replica.extend(replica_set)

                # Here in this for loop, we place all the states and their replicas
                # independently the free capacities of the hosts
                for state in list_of_state_n_replica:  # || pseudocode:36

                    # assigned items can be functions or master states
                    assigned_items = [i[1] for i in assignments if state == i[0]]

                    # assigned_hosts is the list of the hosts of the assigned items
                    assigned_hosts = []
                    for i in assigned_items:
                        if isinstance(i, state_models.Function):
                            assigned_hosts.append(self.get_host_of_function(i))
                        elif isinstance(i, state_models.MasterState):
                            assigned_hosts.append(self.get_node_of_state(i))
                    assigned_hosts = list(dict.fromkeys(assigned_hosts))

                    # Find the optimal destination hosts for master/slave without hosts' capacity check
                    candidate_host, cost, a = self.getOptimalHosts(hosts, assigned_hosts, state, banned_hosts)

                    # Deploy state to the found host
                    self.mapping[candidate_host].load += s.size
                    self.mapping[candidate_host].virtual_elements.append(state)
                    banned_hosts.append(candidate_host)

                    self.print_current_mapping()

            self.print_current_mapping()

        # And now it's time for capacity checking :)
        banned_states_for_state_to_move = []

        ################################################################################################################
        # Pick up the most loaded host
        most_loaded_node = self.get_most_loaded_host()
        # Pick the state what we want to move
        state_to_move = self.get_state_to_move(most_loaded_node, banned_states_for_state_to_move)

        if state_to_move is not None:
            print("State movement is necessary...")
        while state_to_move is not None:
            # mapping_before_move = copy.deepcopy(self.mapping)

            if state_to_move == "error":
                print("State placement failed :(")
                self.print_current_mapping()
                raise Exception("State placement failed :(")

            function_set = self.get_access_functions(state_to_move)
            hosts_of_functions = [h for h in hosts for f in function_set if f in self.topology.nodes[h]['NFs']]

            # print("Picked state: {} \t | Most loaded node: {}".format(state_to_move.id, most_loaded_node))
            self.print_current_mapping()

            # FIXME: is the following method good to give the list of closest_nodes?
            a, b, closest_nodes = self.getOptimalHosts(hosts, hosts_of_functions, state_to_move)
            state_moving_failed = True
            for i in closest_nodes:
                chosen_host = i['node']
                if most_loaded_node != chosen_host:
                    if self.move_is_okay(self.mapping[most_loaded_node], self.mapping[chosen_host],
                                         state_to_move):
                        if isinstance(state_to_move, state_models.MasterState):
                            self.mapping = self.delete_state_from_mapping(state_to_move, most_loaded_node)
                            self.mapping[chosen_host].virtual_elements.append(state_to_move)
                            self.mapping[chosen_host].load += state_to_move.size
                            banned_hosts = [chosen_host]  # we need to save this host, because of the replica moving
                            state_moving_failed = False
                            break
                        elif isinstance(state_to_move, state_models.SlaveReplica) and not self.is_aa_node(
                                state_to_move,
                                self.mapping[
                                    chosen_host]):
                            self.mapping = self.delete_state_from_mapping(state_to_move, most_loaded_node)
                            self.mapping[chosen_host].virtual_elements.append(state_to_move)
                            self.mapping[chosen_host].load += state_to_move.size
                            banned_hosts = []
                            state_moving_failed = False
                            break
            if state_moving_failed:
                banned_states_for_state_to_move.append(state_to_move.id)


            most_loaded_node = self.get_most_loaded_host()
            state_to_move = self.get_state_to_move(most_loaded_node, banned_states_for_state_to_move)

        t2 = datetime.datetime.now()
        self.deploy_states()

        print("FLOODING Heuristic placement finished")
        running_time = t2 - t1

        return running_time

    def place_request_optimally(self, req):

        print("\n### OPTIMAL Placement module #################################################\n")

        if not os.path.isfile(CPLEX_PATH):
            raise RuntimeError('CPLEX does not exist ({})'.format(CPLEX_PATH))

        hosts = self.topology.nodes
        states = req.get_masters()
        replicas = req.get_replicas()

        opt_model = cpx.Model(name="Placement")

        # Binary variables  ###################################################################################
        print("Creating state mapping variables 1...")
        x_vars = {(h, s.id): opt_model.binary_var(name="x{0}{1}".format(h, s.id)) for h in hosts for s in states}

        print("Creating replica mapping variables 2...")
        y_vars = {(i, s.id): opt_model.binary_var(name="y{0}{1}".format(i, s.id)) for i in hosts for s in states}

        # Constraints #########################################################################################
        print("Creating constraints 1 - states can be mapped into only one server")
        for s in states:
            c_name = "c1_{}".format(s.id)
            opt_model.add_constraint(ct=opt_model.sum(x_vars[i, s.id] for i in hosts) == 1, ctname=c_name)

        print("Creating constraints 2 - replicas can be mapped into as many servers as many replicas exist")
        for s in states:
            c_name = "c2_{}".format(s.id)
            # print("State: {}\t Replica number:{}\t Replica instance:{}".format(s.id, s.replica_num,
            #                                                                    len(req.get_replicas_of_state(s))))
            # opt_model.add_constraint(
            #     ct=opt_model.sum(y_vars[i, s.id] for i in hosts) == len(req.get_replicas_of_state(s)), ctname=c_name)
            opt_model.add_constraint(
                ct=opt_model.sum(y_vars[i, s.id] for i in hosts) == s.replica_num, ctname=c_name)

        print("Creating constraints 3 - anti-affinity rules")
        for h in hosts:
            for s in states:
                c_name = "c3_{}_in_{}".format(s.id, h)
                opt_model.add_constraint(ct=(x_vars[h, s.id] + y_vars[h, s.id]) <= 1, ctname=c_name)

        print("Creating constraints 4 - server capacity constraint")
        for h in hosts:
            c_name = "c4_{}".format(h)
            opt_model.add_constraint(
                ct=opt_model.sum((x_vars[h, s.id] + y_vars[h, s.id]) * s.size for s in states) <=
                   self.topology.nodes[h]['capacity'],
                ctname=c_name)

        print("Creating Objective function...")

        def create_w():
            # init w
            w = {(k, s.id): 0 for k in hosts for s in states}
            for s in states:
                for k in hosts:
                    writers = s.writers
                    for i in writers:
                        if MasterController.get_host_of_function(i) == k:
                            w[k, s.id] += i.get_writer_rate_of_state(s)
            return w

        def create_f():
            # init f
            f = {(i, s.id): 0 for i in hosts for s in states}
            for s in states:
                for i in hosts:
                    writers = s.writers
                    for w in writers:
                        if MasterController.get_host_of_function(w) == i:
                            f[i, s.id] = 1
            return f

        def create_g():
            # init g
            g = {(i, s.id): 0 for i in hosts for s in states}
            for s in states:
                for i in hosts:
                    readers = s.readers
                    for r in readers:
                        if MasterController.get_host_of_function(r) == i:
                            g[i, s.id] = 1
            return g

        def create_p():
            # init p
            p = {(i, s.id): 0 for i in hosts for s in states}
            for s in states:
                for i in hosts:
                    readers = s.readers
                    for r in readers:
                        if MasterController.get_host_of_function(r) == i:
                            p[i, s.id] += r.get_reader_rate_of_state(s)
            return p

        w = create_w()
        f = create_f()
        g = create_g()
        p = create_p()

        # ===============================================================================================================
        readers_hosts = dict()
        for s in states:
            readers_hosts.update({s.id: [MasterController.get_host_of_function(r) for r in s.readers]})

        update_cost = sum(sum(
            x_vars[i, s.id] * y_vars[j, s.id] * MasterController.get_min_delay_between_hosts(i, j) for i in hosts for j
            in hosts) * sum(w[k, s.id] for k in hosts) for s in states)

        def get_max_expression(state):

            max_expression = opt_model.max([opt_model.logical_and(x_vars[(i, state.id)], y_vars[
                (j, state.id)]) * MasterController.get_min_delay_between_hosts(i, j) * sum(
                w[(k, state.id)] for k in hosts) for i in hosts for j in hosts if i != j])

            return max_expression

        update_cost = sum(get_max_expression(s) for s in states)
        # update_cost = 0

        write_cost = sum(
            x_vars[i, s.id] * f[j, s.id] * w[j, s.id] * MasterController.get_min_delay_between_hosts(i, j) for i in
            hosts for j in hosts for s in states)

        def delta_cost(host_i, state):
            # FIXME: use a dynamic number instead of the 100 below
            min_delta = opt_model.min((1 - opt_model.logical_or(x_vars[(j, state.id)], y_vars[
                (j, state.id)])) * 100 + MasterController.get_min_delay_between_hosts(host_i, j) for j in hosts)
            return min_delta

        readers_hosts = dict()
        for s in states:
            reader_hosts_list = []
            for r in s.readers:
                host = MasterController.get_host_of_function(r)
                if host not in reader_hosts_list:
                    reader_hosts_list.append(host)
            readers_hosts.update({s.id: reader_hosts_list})

        reading_cost = sum(delta_cost(i, s) * g[(i, s.id)] * p[(i, s.id)] for s in states for i in readers_hosts[s.id])

        quadratic_obj_function = update_cost + write_cost + reading_cost

        # ===============================================================================================================

        opt_model.set_objective("min", quadratic_obj_function)

        print("Exporting the problem")
        cplex_model_path = "{}/cplex_model".format(MasterController.simulation_name)
        opt_model.export_as_lp(basename="cplex_model", path=cplex_model_path)

        t1 = datetime.datetime.now()

        # solving problem in locally
        print("\n\nSolving the problem locally")
        subprocess.call(
            "{} -c 'read {}.lp' 'set timelimit {}' 'set mip interval 1' 'optimize' 'write {}_solution sol'".format(
                CPLEX_PATH, cplex_model_path, CPLEX_TIME_LIMIT, cplex_model_path), shell=True)

        t2 = datetime.datetime.now()

        try:
            with open("{}_solution".format(cplex_model_path)) as myfile:
                head = [next(myfile) for x in range(20)]
                cost = int(round(float(head[6].split('=')[1].replace('\n', '').replace('"', ''))))
            print("\n*** Delay cost: {} ***".format(cost))

        except:
            print("\n\nWARNING: No solution exists!")
            cost = -1

        # indicates whether master state s is placed onto host i
        x_vars = {(h, s.id): 0 for h in hosts for s in states}
        # indicates whether a slave replica of s is placed onto host i
        y_vars = {(i, s.id): 0 for i in hosts for s in states}

        from xml.dom import minidom
        # parse an xml file by name
        mydoc = minidom.parse("{}_solution".format(cplex_model_path))
        variables = mydoc.getElementsByTagName('variable')

        # all item attributes
        print('\nAll variable:')
        for elem in variables:
            if "xhost" in elem.attributes['name'].value:
                if float(elem.attributes['value'].value) > 0.6:
                    host = (elem.attributes['name'].value).split("state")[0][1:]
                    state = "state" + (elem.attributes['name'].value).split("state")[1]
                    x_vars[(host, state)] = 1
            elif "yhost" in elem.attributes['name'].value:
                if float(elem.attributes['value'].value) > 0.6:
                    host = (elem.attributes['name'].value).split("state")[0][1:]
                    state = "state" + (elem.attributes['name'].value).split("state")[1]
                    y_vars[(host, state)] = 1

        running_time = t2 - t1
        print("RUNNING TIME: {}".format(running_time))
        print(x_vars)
        print(y_vars)
        return cost, running_time, x_vars, y_vars

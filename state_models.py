import time
import datetime
import bellmanford as bf
import copy

from enum import Enum
class EventTypes(Enum):
    PLACE_STATES = 1
    ADD_HOST = 2
    DEL_HOST = 3
    FUNCTION_MIGRATION = 4
    DELAY_CHANGE = 5

class Event:
    req = None

    def __init__(self, type, request, hosts, function_distribution, delay_distribution):
        if type not in EventTypes:
            raise Exception(
                "Invalid type '{}' for the event. Please choose among these ones: {}".format(type, [i for i in EventTypes]))
        self.type = type
        self.req = request

class MasterController:
    # class variables
    topology = []
    functions = []
    events = []
    placement_module = None
    replica_controller = None
    running = True
    delay_dist = None
    simulation_name = None

    def __init__(self, topology, functions, placement_module, distribution):

        MasterController.topology = topology
        MasterController.functions = functions
        MasterController.placement_module = placement_module
        MasterController.delay_dist = distribution
        MasterController.min_delay_between_hosts = dict()

        for h1 in topology.nodes:
            for h2 in topology.nodes:
                MasterController.min_delay_between_hosts.update(
                    {(h1, h2): bf.bellman_ford(topology, source=h1, target=h2,
                                               weight="delay")[0]})

    @classmethod
    def get_host_of_function(cls, function):

        if isinstance(function, Function):
            function_id = function.id
        else:
            function_id = function

        try:
            node = next(h for h in cls.topology.nodes if function_id in [i.id for i in cls.topology.nodes[h]['NFs']])
        except StopIteration as e:
            functions = []
            for h in cls.topology.nodes:
                for f in cls.topology.nodes[h]['NFs']:
                    functions.append(f.id)
            print("Function {} is not deployed in the given topology.\nDeployed functions are: {}".format(function_id,
                                                                                                          functions))
            raise e
        return node

    @classmethod
    def get_deployed_states(cls):
        """
        Returns a list of deployed states in the cluster. ATTENTION: only (master) states will be returned, replicas will not!
        :return: list of deployed states in the cluster
        """
        states = []
        for host_id, host in cls.placement_module.mapping.items():
            # ve can be state, replica or function
            for ve in host.virtual_elements:
                if isinstance(ve, MasterState):
                    if ve not in states:
                        states.append(ve)
        return states

    @classmethod
    def get_host_of_state(cls, state):
        for host_id, host in cls.placement_module.mapping.items():
            if state.id in [i.id for i in host.virtual_elements]:
                return host.node

    @classmethod
    def del_function(cls, function):
        if isinstance(function, Function):
            function_id = function.id
        else:
            function_id = function

        host = cls.get_host_of_function(function_id)
        for f in cls.topology.nodes[host]['NFs']:
            if f.id == function_id:
                cls.topology.nodes[host]['NFs'].remove(f)
                break

    @classmethod
    def get_function_object(cls, function_id):
        return next(f for f in cls.functions if f.id == function_id)

    @classmethod
    def get_all_masters(cls):
        masters = []
        for h in list(cls.topology.nodes):
            for m in cls.topology.nodes[h]['states']:
                if isinstance(m, MasterState):
                    if m.id not in [i.id for i in masters]:
                        masters.append(m)
        return masters

    @classmethod
    def get_all_states(cls, only_masters=False):
        """
        Returns a list of deployed states/replicas in the cluster. ATTENTION: if only_masters parameter is True
         then it returns only the (master) states not replicas

        :param only_masters: If it's true, the method returns only the states and not replicas
        :return: list of states/replicas or list of states
        """
        states = []
        for h in list(cls.topology.nodes):
            for m in cls.topology.nodes[h]['states']:
                if only_masters:
                    if isinstance(m, MasterState):
                        if m.id not in [i.id for i in states]:
                            states.append(m)
                else:
                    if m.id not in [i.id for i in states]:
                        states.append(m)
        return states

    @classmethod
    def get_all_replicas(cls):
        """
        This method returns all replica objects which is already deployed
        :return: list of replicas in the cluster
        """
        replicas = [i for i in cls.states if isinstance(i, SlaveReplica)]
        return list(dict.fromkeys(replicas))

    @classmethod
    def check_placement(cls, x_variables = [], y_variables = []):

        hosts = list(cls.topology.nodes)
        states = cls.get_all_states(only_masters=True)

        if x_variables == []:
            # indicates whether master state s is placed onto host i
            x_vars = {(h, s.id): 0 for h in hosts for s in states}
            for s in states:
                host = cls.get_host_of_state(s)
                x_vars[host, s.id] = 1
        else:
            x_vars = x_variables

        if y_variables == []:
            y_vars = {(i, s.id): 0 for i in hosts for s in states}
            for h in cls.topology.nodes:
                for replica in cls.topology.nodes[h]['states']:
                    if 'replica' in replica.id:
                        state_id = 'state{}'.format(replica.id.split('state')[1])
                        y_vars[h, state_id] = 1
        else:
            y_vars = y_variables


        # Constraint 1
        for s in states:
            a = 0
            for h in hosts:
                a += x_vars[h,s.id]
            if not a == 1:
                raise Exception("Constraint 1 is failed for state '{}'".format(s.id))

        # Constraint 2
        for s in states:
            a = 0
            for h in hosts:
                a += y_vars[h, s.id]
            if not a == s.replica_num:
                raise Exception("Constraint 2 is failed. state:{}".format(s.id))

        # Constraint 3
        for s in states:
            for h in hosts:
                a = x_vars[h,s.id] + y_vars[h,s.id]
                if not (a <= 1):
                    raise Exception("Constraint 3 is failed")

        # Constraint 4
        for h in hosts:
            a = 0
            for s in states:
                a += (x_vars[h, s.id] + y_vars[h, s.id])*s.size
            if not (a <= cls.topology.nodes[h]['capacity']):
                raise Exception("Constraint 4 is failed")

        print("CHECKING PLACEMENT FINISHED. EVERY CONSTRAINT IS FULFILLED :)")

        return

    def deploy(self, req, run_opt_too=False):

        def delete_states():
            # host: ID of the host, host_data: All info related to the host
            for host_id in self.topology.nodes:
                host = self.topology.nodes[host_id]
                host['states'] = []

        ### Deploying request  ###########################################

        # 1. Adding slave replica objects to the request

        # If the slave replicas are already existed then delete them
        old_replica_instances = [s for s in req.states if isinstance(s, SlaveReplica)]
        for r in old_replica_instances:
            req.states.remove(r)

        tmp_states = copy.copy(req.states)
        for state in tmp_states:
            req.add_replica_requested_objects(state)

        # 2. Placing states of the request by FLOODING
        heur_running_time = self.placement_module.place_request_flooding(req)
        print("Flooding placement finished at {}".format(datetime.datetime.now()))
        self.check_placement()
        print("Calculating placement cost")
        h_cost = self.calc_cost()
        print("Calculating placement cost finished at {}".format(datetime.datetime.now()))
        print("FLOODING Heuristic cost: {}".format(h_cost))

        cplex_cost = None
        cplex_runtime = None
        if run_opt_too:
            cplex_cost, o_runtime, optimal_x_vars, optimal_y_vars = self.placement_module.place_request_optimally(req)
            self.check_placement(x_variables=optimal_x_vars, y_variables=optimal_y_vars)
            cplex_cost, o_cost_without_update = self.calc_cost(optimal_x_vars, optimal_y_vars)
            print("CPLEX Optimal cost: {}".format(cplex_cost))

        costs = {'optimal': cplex_cost, 'flooding': h_cost}
        runtimes = {'optimal': cplex_runtime, 'flooding': heur_running_time}

        return costs, runtimes


    @classmethod
    def get_placement(cls):

        print("\n*** Current State Placement ***********************************************************")
        for node in list(cls.topology.nodes):
            print("{}: {}\t| {}".format(node, cls.topology.nodes[node]['capacity'] - sum(
                i.size for i in cls.topology.nodes[node]['states']),
                                        [i.id for i in cls.topology.nodes[node]['states']]))

    @classmethod
    def get_data_of_function(cls, function):
        states_of_function = function.writings
        [states_of_function.append(i) for i in function.readings if i not in states_of_function]
        states = [i['state'] for i in states_of_function]
        return states

    @classmethod
    def get_replicas_of_state(cls, state):
        replicas = [i for i in cls.get_all_states() if isinstance(i, SlaveReplica) and i.master == state]
        return replicas

    @classmethod
    def get_min_delay_between_hosts(cls, host1, host2):
        if host1 == host2:
            delay_path = 0
        else:
            delay_path = MasterController.min_delay_between_hosts[(host1, host2)]
        return delay_path

    @classmethod
    def get_xy_values(cls, optimal_x_vars=None, optimal_y_vars=None):

        ################################################################################################################

        hosts = list(cls.topology.nodes)
        states = cls.get_all_states(only_masters=True)

        if optimal_x_vars == None:
            # indicates whether master state s is placed onto host i
            x_vars = {(h, s.id): 0 for h in hosts for s in states}
            for s in states:
                host = cls.get_host_of_state(s)
                x_vars[host, s.id] = 1
        else:
            x_vars = optimal_x_vars

        if optimal_y_vars == None:
            # indicates whether a slave replica of s is placed onto host i
            y_vars = {(i, s.id): 0 for i in hosts for s in states}
            for s in states:
                for h in hosts:
                    data_ids = [r.id for r in cls.topology.nodes[h]['states'] if
                                s.id.split('state')[1] == r.id.split('state')[1] and "replica" in r.id]
                    if data_ids != []:
                        y_vars[h, s.id] = 1

                # replicas = cls.get_replicas_of_state(s)
                # for r in replicas:
                #     host = cls.get_host_of_state(r)
                #     y_vars[host, s.id] = 1
        else:
            y_vars = optimal_y_vars

        return x_vars, y_vars

    @classmethod
    def calc_cost(cls, optimal_x_vars=None, optimal_y_vars=None):

        ################################################################################################################

        hosts = list(cls.topology.nodes)
        states = cls.get_all_states(only_masters=True)

        if optimal_x_vars == None:
            # indicates whether master state s is placed onto host i
            x_vars = {(h, s.id): 0 for h in hosts for s in states}
            for s in states:
                host = cls.get_host_of_state(s)
                x_vars[host, s.id] = 1
        else:
            x_vars = optimal_x_vars

        if optimal_y_vars == None:
            y_vars = {(i, s.id): 0 for i in hosts for s in states}
            for h in cls.topology.nodes:
                for replica in cls.topology.nodes[h]['states']:
                    if 'replica' in replica.id:
                        state_id = 'state{}'.format(replica.id.split('state')[1])
                        y_vars[h, state_id] = 1

        # if optimal_y_vars == None:
        #     # indicates whether a slave replica of s is placed onto host i
        #     y_vars = {(i, s.id): 0 for i in hosts for s in states}
        #     for s in states:
        #         for h in hosts:
        #             data_ids = [r.id for r in cls.topology.nodes[h]['states'] if
        #                         s.id.split('state')[1] == r.id.split('state')[1] and "replica" in r.id]
        #             if data_ids != []:
        #                 y_vars[h, s.id] = 1

        # replicas = cls.get_replicas_of_state(s)
        # for r in replicas:
        #     host = cls.get_host_of_state(r)
        #     y_vars[host, s.id] = 1
        else:
            y_vars = optimal_y_vars

        for h in hosts:
            for s in states:
                if x_vars[h, s.id] == 1 and y_vars[h, s.id] == 1:
                    print("Host: {}, State: {}".format(h, s.id))

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

        readers_hosts = dict()
        for s in states:
            readers_hosts.update({s.id: [MasterController.get_host_of_function(r) for r in s.readers]})

        def logical_and(a, b):
            if a == 1 and b == 1:
                return 1
            else:
                return 0

        def get_max_expression(state):

            max_expression = max([logical_and(x_vars[(i, state.id)], y_vars[
                (j, state.id)]) * MasterController.get_min_delay_between_hosts(i, j) * sum(
                w[(k, state.id)] for k in hosts) for i in hosts for j in hosts if i != j])

            return max_expression

        update_cost = sum(get_max_expression(s) for s in states)
        # update_cost = 0

        write_cost = sum(
            x_vars[i, s.id] * f[j, s.id] * w[j, s.id] * MasterController.get_min_delay_between_hosts(i, j) for i in
            hosts for j in hosts for s in states)

        # def delta_cost(host_i, state):
        #     # FIXME: get the max delay and add 100000 to it instead of the 1000000 value below
        #     min_delta = min((1 - (x_vars[(j, state.id)] + y_vars[
        #         (j, state.id)])) * 1000000 + MasterController.get_min_delay_between_hosts(host_i, j) for j in hosts)
        #
        #     return min_delta

        def logical_or(a, b):
            if a == 1:
                return 1
            if b == 1:
                return 1
            else:
                return 0

        def delta_cost(host_i, state):
            # FIXME: get the max delay and add 100000 to it instead of the 1000000 value below
            min_delta = min((1 - logical_or(x_vars[(j, state.id)], y_vars[
                (j, state.id)])) * 1000000 + MasterController.get_min_delay_between_hosts(host_i, j) for j in hosts)
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

        quadratic_obj_function_cost = update_cost + write_cost + reading_cost

        print("Update: {}".format(update_cost))
        print("Write: {}".format(write_cost))
        print("Reading: {}".format(reading_cost))

        ################################################################################################################

        return quadratic_obj_function_cost

    def run(cls, run_opt_too):

        costs = None
        runtimes = None

        while (cls.running):

            cls.get_placement()
            cls.check_placement()

            if cls.events == []:
                time.sleep(1)
                print('--> Nothing to do...')
                return costs, runtimes

            else:
                e = cls.events.pop(0)

                # if event is (a) service deploying
                if e.type == EventTypes.PLACE_STATES:
                    print("--> Placing states")
                    costs, runtimes = cls.deploy(e.req, run_opt_too)

                #  if the event is host number changing (new host or deleted host)
                if e.type == EventTypes.ADD_HOST:
                    print("--> ADDING HOSTS")
                    for h in e.hosts:
                        print("Adding host {}".format(h['hostID']))
                        cls.topology.add_node(h['hostID'], capacity=h['capacity'], NFs=[], states=[])
                if e.type == EventTypes.DEL_HOST:
                    print("--> DELETING HOSTS")
                    for h in e.hosts:
                        print("Deleting host {}".format(h))
                        cls.topology.remove_node(h)

                # if event is function distribution change
                if e.type == EventTypes.FUNCTION_MIGRATION:
                    print("--> REACTING FOR FUNCTION PLACEMENT CHANGE")
                    moved_functions = []
                    masters_to_replace = []
                    for function_id, host in e.function_mapping.items():
                        if cls.get_host_of_function(function_id) != host:
                            cls.del_function(function_id)
                            print("Deleting function {}".format(function_id))
                            function = cls.get_function_object(function_id)
                            cls.topology.nodes[host]['NFs'].append(function)
                            moved_functions.append(function)

                    for f in moved_functions:
                        for i in f.writings:
                            if i['state'] not in masters_to_replace:
                                masters_to_replace.append(i['state'])
                        for i in f.readings:
                            if i['state'] not in masters_to_replace:
                                masters_to_replace.append(i['state'])

                    print('Replacing the following states: {}'.format([i.id for i in masters_to_replace]))
                    if masters_to_replace != []:
                        req = Request(masters_to_replace)
                        cls.deploy(req)

                # if event is delay distribution change
                if e.type == EventTypes.DELAY_CHANGE:
                    print("--> REACTING FOR DELAY DISTRIBUTION CHANGE")
                    cls.delay_dist = e.distribution
                    masters = cls.get_all_masters()
                    print('Replacing the following states: {}'.format([i.id for i in masters]))
                    req = Request(masters)
                    cls.deploy(req)


class Function:
    def __init__(self, id, readings, writings):
        self.id = 'function{}'.format(id)
        self.readings = readings
        self.writings = writings

    def add_state_to_read(self, state, rate):
        self.readings.append({'state': state, 'rate': rate})
        state.readers.append(self)

    def add_state_to_write(self, state, rate):
        self.writings.append({'state': state, 'rate': rate})
        state.writers.append(self)

    def get_writer_rate_of_state(self, state):
        rate = next(i['rate'] for i in self.writings if i['state'].id == state.id)
        return rate

    def get_reader_rate_of_state(self, state):
        rate = next(i['rate'] for i in self.readings if i['state'].id == state.id)
        return rate


class MasterState:
    def __init__(self, id, size, slave_num):
        self.id = 'state{}'.format(id)
        self.size = size
        self.replica_num = slave_num
        self.user_defined_slave_num = slave_num
        self.readers = []
        self.writers = []

    def get_read_only_functions(self, ):
        return [r for r in self.readers if r not in self.writers]


class SlaveReplica:
    def __init__(self, id, size, state, readers):
        self.id = 'replica{}'.format(id)
        self.size = size
        self.master = state
        self.readers = readers

class Request:
    def __init__(self, states):
        self.states = states  # here the states can be both (master) state and replica
        self.slaves_of_states = dict()

    def get_replicas_of_state(self, state):
        try:
            replicas = self.slaves_of_states[state.id]
            return replicas
        except Exception as e:
            for d in self.states:
                if isinstance(d, SlaveReplica):
                    try:
                        self.slaves_of_states[d.master.id].append(d)
                    except Exception:
                        self.slaves_of_states.update({d.master.id: []})
                        self.slaves_of_states[d.master.id].append(d)
                else:
                    try:
                        self.slaves_of_states[d.master.id]
                    except Exception:
                        self.slaves_of_states.update({d.id: []})

            return self.slaves_of_states[state.id]

    def get_masters(self):
        masters = [i for i in self.states if isinstance(i, MasterState)]
        return list(dict.fromkeys(masters))

    def get_replicas(self):
        """
        This method returns all replica objects in the request
        :return: list of replicas in the request
        """
        replicas = [i for i in self.states if isinstance(i, SlaveReplica)]
        return list(dict.fromkeys(replicas))

    def get_functions(self):
        functions = []
        for s in self.states:
            if isinstance(s, MasterState):
                functions.extend(s.readers)
                functions.extend(s.writers)
        return list(dict.fromkeys(functions))

    def get_slaves(self):
        slaves = [i for i in self.states if isinstance(i, SlaveReplica)]
        return list(dict.fromkeys(slaves))

    def add_replica_requested_objects(self, state):
        for i in range(state.replica_num):
            self.states.append(SlaveReplica("{}_of_{}".format(i, state.id), state.size, state, state.readers))
        return


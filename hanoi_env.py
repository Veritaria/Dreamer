import gym 
from gym import Wrapper, spaces
import numpy as np 
import random
import itertools
import numpy as np
import networkx as nx
import itertools
import matplotlib.patches as patches
import matplotlib.pyplot as plt 
def fig2data(fig, dpi: int = 70):
    #fig.set_dpi(dpi)
    fig.set_figwidth(5)
    fig.set_figheight(5)
    fig.canvas.draw()

    # copy image data from buffer
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).copy()

    # get the dpi adjusted figure dimensions
    width, height = map(int, fig.get_size_inches() * fig.get_dpi())
    data = data.reshape(height, width, 4)
    
    data[..., [0, 1, 2, 3]] = data[..., [1, 2, 3, 0]]

    return data


def get_peg_params(pegs, width, height):
    peg_width = (width / 10.) / len(pegs)
    vertical_padding = height * 0.2
    peg_height = height - vertical_padding
    boundaries = np.linspace(0, width, len(pegs)+1)
    interval = (boundaries[1] - boundaries[0]) / 2
    peg_midpoints = boundaries[:-1] + interval
    peg_to_hor_midpoints = dict(zip(pegs, peg_midpoints))
    return peg_width, peg_height, peg_to_hor_midpoints

def get_disc_params(discs_ordered_by_size, peg_to_disc_list, peg_to_hor_midpoints, width, peg_height):
    num_pegs = len(peg_to_hor_midpoints)
    num_discs = len(discs_ordered_by_size)
    disc_height = (peg_height * 0.75) / num_discs

    horizontal_padding = width * 0.1
    max_disc_width = width / num_pegs - horizontal_padding
    min_disc_width = max_disc_width / 3
    all_disc_widths = np.linspace(max_disc_width, min_disc_width, num_discs)
    disc_widths = dict(zip(discs_ordered_by_size, all_disc_widths))

    disc_midpoints = {}
    for peg, discs in peg_to_disc_list.items():
        x = peg_to_hor_midpoints[peg]
        for i, disc in enumerate(discs):
            y = i * disc_height + disc_height / 2
            disc_midpoints[disc] = (x, y)

    return disc_height, disc_midpoints, disc_widths

def draw_pegs(ax, peg_width, peg_height, peg_to_hor_midpoints, height):
    for midx in peg_to_hor_midpoints.values():
        x = midx - peg_width / 2
        y = 0
        rect = patches.Rectangle((x,y), peg_width, peg_height, 
            linewidth=1, edgecolor=(0.2,0.2,0.2), facecolor=(0.5,0.5,0.5))
        ax.add_patch(rect)

def draw_discs(ax, disc_height, disc_midpoints, disc_widths):
    for disc, (midx, midy) in disc_midpoints.items():
        disc_width = disc_widths[disc]
        x = midx - disc_width / 2
        y = midy - disc_height / 2
        rect = patches.Rectangle((x,y), disc_width, disc_height, 
            linewidth=1, edgecolor=(0.2,0.2,0.2), facecolor=(0.8,0.1,0.1))
        ax.add_patch(rect)



class HanoiEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, size=4,n_disks=3, env_noise=0,begin_noise=0,test=False):
        assert n_disks<=size 
        self.n_disks = n_disks
        self.size=size 
        self.env_noise = env_noise
        self.begin_noise = begin_noise 
        self.action_space = spaces.Discrete(6)
        self.test=test 
        # self.observation_space = spaces.Discrete(3**self.num_disks)
        self.observation_space = spaces.MultiDiscrete([3 for _ in range(self.size)])
        # self.observation_space = spaces.Tuple(self.num_disks*(spaces.Discrete(3),))

        self.current_state = None
        self.goal_state=np.array([2 for _ in range(self.size)])

        self.done = None
        self.ACTION_LOOKUP = {0 : "(0,1) - top disk of pole 0 to top of pole 1 ",
                              1 : "(0,2) - top disk of pole 0 to top of pole 2 ",
                              2 : "(1,0) - top disk of pole 1 to top of pole 0",
                              3 : "(1,2) - top disk of pole 1 to top of pole 2",
                              4 : "(2,0) - top disk of pole 2 to top of pole 0",
                              5 : "(2,1) - top disk of pole 2 to top of pole 1"}
        self.action_to_move = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

        self.G = self.get_movability_map()
        self.all_subgoals = self.get_all_subgoals()

    def step(self, action):

        """
        * Inputs:
            - action: integer from 0 to 5 (see ACTION_LOOKUP)
        * Outputs:
            - current_state: state after transition
            - reward: reward from transition
            - done: episode state
            - info: dict of booleans (noisy?/invalid action?)
        0. Check if transition is noisy or not
        1. Transform action (0 to 5 integer) to tuple move - see Lookup
        2. Check if move is allowed
        3. If it is change corresponding entry | If not return same state
        4. Check if episode completed and return
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        info = {"transition_failure": False,
                "invalid_action": False}

        if self.env_noise > 0:
            r_num = random.random()
            if r_num <= self.env_noise:
                action = random.randint(0, self.action_space.n-1)
                info["transition_failure"] = True
        move = self.action_to_move[action]

        if self.move_allowed(move): 
            disk_to_move = min(self.disks_on_peg(move[0]))
            self.current_state[disk_to_move] = move[1]
        else:
            info["invalid_action"] = True

        if np.equal(self.current_state, self.goal_state).all():
            reward = 10.0
            self.done = True
        elif info["invalid_action"] == True:
            reward = -1.0
        else:
            reward = -0.1

        truncated = False

        return self.current_state, reward, self.done,  info

    def convert_input_state(self, state):
        """
        convert input state (int) to 3-dim vectors
        """
        converted_state = np.zeros(self.size)
        for i in range(self.size):
            converted_state[i] = state % 3
            state = int((state - converted_state[i]) / 3)
        return converted_state

    def convert_output_state(self, state):
        """
        convert output state (3-dim vectors) to int
        """
        converted_state = 0
        for i in range(self.size):
            converted_state += state[i] * 3**i
        return converted_state

    def disks_on_peg(self, peg):
        """
        * Inputs:
            - peg: pole to check how many/which disks are in it
        * Outputs:
            - list of disk numbers that are allocated on pole
        """
        return [disk for disk in range(self.size) if self.current_state[disk] == peg]

    def move_allowed(self, move):
        """
        * Inputs:
            - move: tuple of state transition (see ACTION_LOOKUP)
        * Outputs:
            - boolean indicating whether action is allowed from state!
        move[0] - peg from which we want to move disc
        move[1] - peg we want to move disc to
        Allowed if:
            * discs_to is empty (no disc of peg) set to true
            * Smallest disc on target pole larger than smallest on prev
        """
        disks_from = self.disks_on_peg(move[0])
        disks_to = self.disks_on_peg(move[1])

        if disks_from:
            return (min(disks_to) > min(disks_from)) if disks_to else True
        else:
            return False

    def reset(self, seed=None, options=None):
        self.current_state=np.array([0 for _ in range(self.size)])
        if self.size>self.n_disks:
            for i in range(self.n_disks,self.size):
                self.current_state[i]=2
        if self.begin_noise!=0:
            if self.test:
                for i in range(self.size):
                    self.current_state[i]=np.random.choice([0,1,2])
            else:
                for i in range(self.n_disks):
                    self.current_state[i]=np.random.choice([0,1,2]) 

        if self.size==self.n_disks:
            self.current_state[-1]=0 
            
        self.done = False
        info = {"transition_failure": False, "invalid_action": False}
        return self.current_state

    def render(self, mode='human', close=False):
        return

    def set_env_parameters(self, n_disks=4, env_noise=0, begin_noise=0,test=False,verbose=True):
        self.n_disks = n_disks
        self.env_noise = env_noise
        self.begin_noise=begin_noise     
        self.test=test    
        # self.observation_space = spaces.Discrete(3**self.num_disks)

        self.goal_state=np.array([2 for _ in range(self.size)])
        self.G = self.get_movability_map()

        if verbose:
            print("Hanoi Environment Parameters have been set to:")
            print("\t Number of Disks: {}".format(self.n_disks))
            print("\t Transition Failure Probability: {}".format(self.env_noise)) 
            print('\t Noisy Start State: {}'.format(self.begin_noise))

    def get_movability_map(self):
        """
        nodes of graph: all possible states, each represented by (x1, x2, ..., xn), xi = 0, 1, 2, is the peg number of disk i
        edges of graph: all possible transitions, each represented by (state1, state2), which means moving from state1 to state2
            edge attriute: "action", represented by (x, y), which means moving the top disk of peg x to the top of peg y, x, y = 0, 1, 2
        """
        G = nx.DiGraph()
        states = list(itertools.product(range(3), repeat=self.size))
        moves = list(itertools.permutations(list(range(3)), 2))
        G.add_nodes_from(itertools.product(range(3), repeat=self.size))

        for state in states:
            for move in moves:
                current_peg = move[0]
                target_peg  = move[1]
                # Indices of plates on current peg
                discs_from = np.where(np.array(state) == current_peg)[0]
                # Indices of plates on target peg
                discs_to   = np.where(np.array(state) == target_peg)[0]
                if discs_from.size == 0:
                    continue
                if discs_to.size == 0 or np.min(discs_to) > np.min(discs_from):
                    state_to = list(state)
                    state_to[discs_from[0]] = target_peg
                    G.add_edge(state, tuple(state_to), action=move)
        
        betweenness_centrality = nx.betweenness_centrality(G)
        for node in G.nodes:
            G.nodes[node]['betweenness_centrality'] = betweenness_centrality[node]
        
        return G
    
    def get_shortest_path(self, source, target):
        """
        get shortest path from source state to target state, 
        input:
            source and target: represented by (x1, x2, ..., xn), xi = 0, 1, 2, is the peg number of disk i
        return:
            1. a list of all shortes paths, each path is a list of states
            2. a list of nodes with the max centrality in the shortest path (not including the source)
        """
        sp = nx.all_shortest_paths(self.G, source, target)
        sp = list(sp)
        max_centrality = [0 for _ in range(len(sp))]
        max_centrality_nodes = [[] for _ in range(len(sp))]
        for i, p in enumerate(sp):
            for n in p[1:]:
                if self.G.nodes[n]['betweenness_centrality'] > max_centrality[i]:
                    max_centrality[i] = self.G.nodes[n]['betweenness_centrality']
                    max_centrality_nodes[i] = [n]
                elif self.G.nodes[n]['betweenness_centrality'] == max_centrality[i]:
                    max_centrality_nodes[i].append(n)
        return sp, max_centrality_nodes
    
    def get_current_subgoal(self):
        """ 
        get the state with max centrality in the shortest path from current state to goal state
        return:
            np.array of the subgoal state, the same format as self.current_state
        """
        _, max_centrality_nodes = self.get_shortest_path(tuple(self.current_state), tuple(self.goal_state))
        if len(max_centrality_nodes) > 0 and len(max_centrality_nodes[0]) > 0 and max_centrality_nodes[0][0] in self.all_subgoals:
            return np.array(max_centrality_nodes[0][0])
        else:
            return np.array(self.goal_state)
    
    def get_all_subgoals(self):
        """
        get all first-hierarchy subgoals in current path, i.e. the nodes with max centrality in the whole graph, usually 6 nodes
        """
        subgoals = []
        max_centrality = 0
        for node in self.G.nodes:
            if self.G.nodes[node]['betweenness_centrality'] > max_centrality:
                max_centrality = self.G.nodes[node]['betweenness_centrality']
        for node in self.G.nodes:
            if self.G.nodes[node]['betweenness_centrality'] == max_centrality:
                subgoals.append(node)
        return subgoals

    def get_length_optimal_trajectory(self):
        path,subgoal_states=self.get_shortest_path(tuple(self.current_state),tuple(self.goal_state))
        return len(path[0])
    def get_optimal_trajectory(self):
        trajectory=[]
        path,subgoal_states=self.get_shortest_path(tuple(self.current_state),tuple(self.goal_state))
        
        subgoal_states=subgoal_states[0]
        path=path[0]
        subgoal_states.append(tuple(self.goal_state))
        subgoal_distances={}

        for subgoal in subgoal_states:
            dist=0
            if subgoal in path:
                while path[dist]!=subgoal:
                    dist+=1
                if dist>0:
                    subgoal_distances[subgoal]=dist 
        subgoals_ordered=sorted(list(subgoal_distances.keys()),key=lambda x: subgoal_distances[x])
        curr_subgoal=subgoals_ordered[0]
        for idx in range(len(path)):
            curr_state=path[idx]
            if curr_state==tuple(self.goal_state):
                assert len(subgoals_ordered)==1 and subgoals_ordered[0]==tuple(self.goal_state)
                trajectory.append({
                    'state':curr_state,
                    'action':6,
                    'reward':10,
                    'subgoal':self.goal_state 
                })
            else:
                if curr_state==curr_subgoal:
                    subgoals_ordered.remove(curr_subgoal)
                    curr_subgoal=subgoals_ordered[0]
                next_state=path[idx+1]
                next_action=self.G.get_edge_data(curr_state,next_state)['action']
                trajectory.append({
                    'state':curr_state,
                    'action':self.action_to_move.index(next_action),
                    'reward':-0.1,
                    'subgoal':curr_subgoal
                })
        return trajectory 
    def render(self, mode='human', close=False): 
        obs=self.current_state[::-1]
        width, height = 4.2, 1.5
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0),
                                    aspect='equal', frameon=False,
                                    xlim=(-0.05, width + 0.05),
                                    ylim=(-0.05, height + 0.05))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(plt.NullFormatter())
            axis.set_major_locator(plt.NullLocator())
        
        pegs=list(range(3))
        discs_ordered_by_size=list(range(self.size))
        peg_to_disc_list={i:[] for i in pegs}
        for d,p in enumerate(obs):
            peg_to_disc_list[p].append(d)
        peg_width, peg_height, peg_to_hor_midpoints = get_peg_params(pegs, width, height)
        disc_height, disc_midpoints, disc_widths = get_disc_params(discs_ordered_by_size, 
            peg_to_disc_list, peg_to_hor_midpoints, width, peg_height)

        draw_pegs(ax, peg_width, peg_height, peg_to_hor_midpoints, height)
        draw_discs(ax, disc_height, disc_midpoints, disc_widths)

        return fig2data(fig)
    
def register_hanoi_env(env_id="Hanoi-v0", n_disks=3,size=4, env_noise=0, begin_noise=0, max_episode_steps=50,test=False):
    gym.envs.register(id=env_id,entry_point=HanoiEnv, max_episode_steps=max_episode_steps, kwargs={'n_disks':n_disks,'size':size,'env_noise':env_noise, 'begin_noise':begin_noise,'test':test})

register_hanoi_env(env_id='hanoi-v0',n_disks=3,size=4,begin_noise=1,test=False)
register_hanoi_env(env_id='hanoitest-v0',n_disks=3,size=4,begin_noise=1,test=True)

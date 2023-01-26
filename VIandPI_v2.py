"""
Compatible to discrete_env.py
"""

import numpy as np
import copy

class Value_iteration():
    def __init__(self, env, discount_rate):
        self.n_states = env.nS  # total number of states
        self.a_size = env.nA
        self.discount_rate = discount_rate
        self.transition_matrix = env.P
        self.v_table = np.zeros(self.n_states)

    def solve(self, max_iter = 1000, theta = 1.0e-8):
        policy = np.zeros(self.n_states)
        new_v = np.zeros(self.n_states)
        Q_sa = np.zeros(self.a_size)
        for iter in range(max_iter):
            if iter%100 == 0:
                print(iter)
                print(self.v_table)
            for s in range(self.n_states):
                for a in range(self.a_size):
                    possible_transitions = self.transition_matrix[s][a] # (prob, n_s, reward, done)
                    Q_sa[a] = 0
                    for t in enumerate(possible_transitions):
                        Q_sa[a] += t[1][0] * (t[1][2] + self.discount_rate * self.v_table[t[1][1]] * (1.0 - t[1][3]))
                    # for j in range(len(possible_transitions)):
                    #     Q_sa[a] += possible_transitions[j][0] * (
                    #                 possible_transitions[j][2] + self.discount_rate * self.v_table[
                    #             possible_transitions[j][1]] * (1.0 - possible_transitions[j][3]))
                new_v[s] = np.max(Q_sa)
                policy[s] = np.argmax(Q_sa)

            max_delta = np.max(np.abs(new_v - self.v_table))
            self.v_table = copy.deepcopy(new_v)

            # print(iter, max_delta)
            if max_delta < theta:
                break
        return policy, self.v_table  # return policies and optimal values

class Policy_iteration():
    def __init__(self, env, discount_rate):
        self.n_states = env.nS  # total number of states
        self.a_size = env.nA
        self.discount_rate = discount_rate
        self.transition_matrix = env.P

    def solve(self, max_iter=1000, theta=1.0e-8):
        self.policy = np.zeros(self.n_states)
        converged = False
        policy_steps = 1
        while not converged:
            print("Policy steps: ", policy_steps)

            policy_value = self.iterative_policy_evaluation(self.policy, max_iter, theta)
            new_policy = self.greedy_policy_improvement(policy_value)

            if np.all(np.equal(self.policy, new_policy)):
                converged = True
            self.policy = copy.deepcopy(new_policy)
            policy_steps += 1

        return self.policy, policy_value

    def greedy_policy_improvement(self, policy_value):
        new_policy = np.zeros(self.n_states)
        Q_sa = np.zeros(self.a_size)
        for s in range(self.n_states):
            for a in range(self.a_size):
                possible_transitions = self.transition_matrix[s][a] # (prob, n_s, reward, done)
                Q_sa[a] = 0
                for t in enumerate(possible_transitions):
                    Q_sa[a] += t[1][0] * (t[1][2] + self.discount_rate * policy_value[t[1][1]] * (1.0 - t[1][3]))
            new_policy[s] = np.argmax(Q_sa)

        return new_policy

    def iterative_policy_evaluation(self, policy, max_iter=1000, theta=1.0e-8):
        policy_value = np.zeros(self.n_states)
        prev_value = np.zeros(self.n_states)
        for iter in range(max_iter):
            for s in range(self.n_states):
                a = int(policy[s])
                possible_transitions = self.transition_matrix[s][a]
                policy_value[s] = 0
                for t in enumerate(possible_transitions):
                    policy_value[s] += t[1][0] * (t[1][2] + self.discount_rate * prev_value[t[1][1]] * (1.0 - t[1][3]))

            max_delta = np.max(np.abs(policy_value - prev_value))
            prev_value = copy.deepcopy(policy_value)
            # print(iter, max_delta)
            if max_delta < theta:
                break
        return policy_value  # return policy values
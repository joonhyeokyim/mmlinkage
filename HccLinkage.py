import torch
import numpy as np
import networkx as nx
import random

class HccLinkage():

    def __init__(self, d, alt = False, rand = False, tol = 1e-5):
        if type(d) == np.ndarray:
            self.d = torch.tensor(d)
        else:
            self.d = d
        self.n = self.d.shape[0]
        self.alt = alt
        self.rand = rand
        self.tol = tol
        self.nextroots = list(range(self.n,2*self.n))
        self.nextroots.reverse()
        self.d_U = torch.zeros(self.n, self.n, dtype = torch.double)
        self.A = torch.zeros(self.n, self.n, dtype = torch.int)
        self.N = torch.zeros(self.n, 2 * self.n, dtype = torch.int)
        self.H = torch.zeros(self.n, 2 * self.n, dtype = torch.int)
        self.M = torch.zeros(2 * self.n, 2 * self.n, dtype = torch.int)
        self.S = torch.ones(2 * self.n, dtype = torch.int)
        # self.set_of_clusters = set(range(self.n))
        self.membership = torch.arange(self.n, dtype = torch.int)
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n))
        self.debug = False

    
    def get_edge_seq(self, d, n, rand = False):
        entries = []
        edges = []
        for i in range(n):
            for j in range(i):
                entries.append((i, j, d[i,j]))
        if rand:
            random.shuffle(entries)
        entries.sort(key = lambda e: e[2])
        for e in entries:
            edges.append((e[0],e[1]))
        # print(edges)
        return edges

    def update_matrices(self, i, j):
        k = self.membership[i]
        l = self.membership[j]
        self.A[i, j] += 1
        self.A[j, i] += 1
        self.N[i, l] += 1
        self.N[j, k] += 1
        if k != l:
            if self.H[i, l] == 0 and 2 * self.N[i, l] >= self.S[l]:
                self.H[i, l] = 1
                self.M[k, l] += 1
            if self.H[j, k] == 0 and 2 * self.N[j, k] >= self.S[k]:
                self.H[j, k] = 1
                self.M[l, k] += 1

    def merge_clusters(self, k, l, distance):
        # print("added ", k, l)
        r = self.nextroots.pop(-1)
        new_size = self.S[k] + self.S[l]
        self.S[r] = new_size
        self.N[:, r] = self.N[:, k] + self.N[:, l]
        X = []
        Y = []
        for v in range(self.n):
            if 2 * self.N[v, r] >= new_size:
                self.H[v, r] = 1
                self.M[self.membership[v], r] += 1
            if self.membership[v] == k:
                # self.membership[v] = r
                X.append(v)
            if self.membership[v] == l:
                # self.membership[v] = r
                Y.append(v)
        self.M[r, :] = self.M[k, :] + self.M[l, :]
        for x in X:
            for y in Y:
                self.d_U[x,y] = distance
                self.d_U[y,x] = distance
        # print("added ", k, l)
        for x in X:
            self.membership[x] = r
        for y in Y:
            self.membership[y] = r
        if self.debug:
            print(X, Y, distance.item())
        # self.set_of_clusters.remove(k)
        # self.set_of_clusters.remove(l)
        # self.set_of_clusters.add(r)


    def learn_UM(self):
        E = self.get_edge_seq(self.d, self.n)
        t = 0
        while(len(self.nextroots) > 1):
            i, j = E[t][0], E[t][1]
            self.update_matrices(i, j)
            k, l = self.membership[i], self.membership[j]
            if(k != l and self.M[k,l] + self.M[l,k] == self.S[k] + self.S[l]):
                self.merge_clusters(k, l, self.d[i,j])
                # if(self.alt):
            # print(t)
            t += 1


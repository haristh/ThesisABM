#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:02:55 2023

@author: charistheodorou
"""

import mesa
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import numpy as np
import random
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

#%% Functions


# Specify the path of the Excel file and the name of the worksheet
excel_file_path = 'CHISdata.xlsx'


# Read the data from the worksheet into a dataframe
country_df = pd.read_excel(excel_file_path, sheet_name='Countries', header=0, index_col='Country')
sector_df = pd.read_excel(excel_file_path, sheet_name='Sectors', header=0, index_col='Sector')



country_probs = country_df[['Percentage']].iloc[0:56]
sector_probs = sector_df[['Percentage']].iloc[0:13]





#%% Functions


class CompanyAgent(mesa.Agent):
    """An agent representing a company that joins SBTi."""

    def __init__(self, unique_id, model, country_probs, sector_probs):
        super().__init__(unique_id, model)
        
        #PARAMETERS
        
        self.country = self.choose_country(country_probs)
        self.sector = self.choose_sector(sector_probs)
        #self.culturechar = None
        #self.emissions = None
        self.internal_target = np.random.rand()
        #self.shareholdernumber = None
        
        
        
        #VARIABLES
        
        self.list_connections = None
        
        #processes
        self.social_pressure = None
        self.motivation = None
        self.target_progress = None
        
        #states of agent
        self.is_aware = False
        self.is_committed = False
        self.has_target = False
        
        # Generate a random number from a uniform distribution in the range [0, 1)
        self.leadership = np.random.rand()
        self.riskawareness= np.random.rand()
        self.reputation= np.random.rand()

        
        
        

    def choose_country(self, country_probs):
        """Selects a country based on the given probabilities."""
        countries = country_probs.index.tolist()
        probs = country_probs.Percentage.tolist()
        return self.random.choices(countries, probs)[0]

    def choose_sector(self, sector_probs):
        """Selects a sector based on the given probabilities."""
        sectors = sector_probs.index.tolist()
        probs = sector_probs.Percentage.tolist()
        return self.random.choices(sectors, probs)[0]
    
    
    def step(self):
        if not self.is_aware:
            # check if company becomes aware of SBTi
            pass  # code to determine if company becomes aware of SBTi
        elif self.is_aware and not self.is_committed:
            # check if company commits to SBTi
            pass  # code to determine if company commits to SBTi
        elif self.is_aware and self.is_committed and not self.has_target:
            # check if company sets target with SBTi
            pass  # code to determine if company sets target with SBTi
        else:
            # company has awareness, commitment, and target set with SBTi
            pass  # code for companies with awareness, commitment, and target set



class SBTiModel(mesa.Model):
    def __init__(self, num_companies, country_probs, sector_probs, seed=None):
        self.num_companies = num_companies
        self.schedule = RandomActivation(self)
        
        # define weights for connectivity matrix
        self.alpha = 0.7
        self.beta = 0.3
        
        self.agents = []
        
#        self.grid = NetworkGrid({})
        
        # add nodes for companies to network
        for i in range(self.num_companies):
             a = CompanyAgent(i, self, country_probs, sector_probs)
             self.schedule.add(a)
             self.agents.append(a)
        #     self.grid.add_node(c)
            
        
        # create initial network based on connectivity matrix M
        self.M = self.generateM()
        self.G = self.generateNetwork()
        
        # parameters for the Watts-Strogatz network
        self.avg_degree = 6
        self.rewiring_prob = 0.3
        
        # seed for random number generation
        if seed:
            random.seed(seed)
            np.random.seed(seed)      
    
    # create connectivity matrix M
    def generateM(self):
        M = np.zeros((self.num_companies, self.num_companies))
        for i in range(self.num_companies):
            for j in range(self.num_companies):
                if i == j:
                    M[i, j] = 0  # no self-loops
                    
                else:
                    if self.schedule.agents[i].sector == self.schedule.agents[j].sector:
                        M[i, j] += self.alpha
                    if self.schedule.agents[i].country == self.schedule.agents[j].country:
                        M[i, j] += self.beta
                #print(self.M[i, j])
        return M
    
    def generateNetwork(self):
        # create network based on connectivity matrix M
        G = nx.DiGraph()
        
        for i in range(self.num_companies):
            for j in range(self.num_companies):
                if i != j and random.random() < self.M[i][j]:
                    G.add_edge(self.agents[i], self.agents[j])
        return G
        # create Watts-Strogatz network
        #self.G = nx.watts_strogatz_graph(self.num_companies, 6, 0.3)

    def step(self):
        self.schedule.step()
        # update the network using the Watts-Strogatz algorithm
        self.G = nx.watts_strogatz_graph(self.num_companies, self.avg_degree, self.rewiring_prob, seed=self.schedule.steps)

        # remove edges from the old network that are not present in the new network
        self.G.remove_edges_from([(u, v) for (u, v) in self.G.edges if (u not in self.agents or v not in self.agents)])

        # add edges from the new network that are not present in the old network
        self.G.add_edges_from([(self.agents[i], self.agents[j]) for i, j in self.G.edges if (self.agents[i], self.agents[j]) not in self.G.edges])

        # update the step count
        self.schedule.steps += 1

        # step through the agents
        self.schedule.step()

        # plot the new network
        #fig, ax = plt.subplots(figsize=(10, 10))
        #pos = nx.spring_layout(self.G)
        #nx.draw(self.G, pos=pos,node_size=300, with_labels=False)
        #ax.set_title(f"Network at Step {self.schedule.steps}")
        #plt.show()
    



model = SBTiModel(5, country_probs, sector_probs, seed = 2)
#M = model.generateM()
#agents = model.schedule

# for i in range(10):
#     # Generate the new network
#     model.step()
#     new_pos = nx.spring_layout(model.G)

#     # Plot the new network
#     fig, ax = plt.subplots(figsize=(10, 10))

#     nx.draw_networkx_nodes(model.G, pos=new_pos, node_size=50)
#     nx.draw_networkx_edges(model.G, pos=new_pos)
#     ax.set_title("Network at Step {i}")
#    plt.show()  # display the plot in a separate window



# plot the initial network
fig, ax = plt.subplots(figsize=(10, 10))
pos = nx.spring_layout(model.G)
nx.draw(model.G, pos=pos, with_labels=True, node_size=300)
ax.set_title("Initial Network")




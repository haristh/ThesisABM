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

from mesa.visualization.modules import NetworkModule
from mesa.visualization.ModularVisualization import ModularServer


#%% Functions


# Specify the path of the Excel file and the name of the worksheet
excel_file_path = 'CHISdata.xlsx'
excel_file_path2 = 'CountriesData.xlsx'

# Read the data from the worksheet into a dataframe

sector_df = pd.read_excel(excel_file_path, sheet_name='Sectors', header=0, index_col='Sector')
country_df = pd.read_excel(excel_file_path2, sheet_name='Countries', header=0, index_col='Country')



sector_probs = sector_df[['Percentage']].iloc[0:13]
country_probs = country_df[['Percentage']].iloc[0:49]


mu_communicating = country_df[['Communicating']].iloc[0:49]
mu_evaluating = country_df[['Evaluating']].iloc[0:49]
mu_leading = country_df[['Leading']].iloc[0:49]
mu_deciding = country_df[['Deciding']].iloc[0:49]
mu_trusting = country_df[['Trusting']].iloc[0:49]
mu_disagreeing = country_df[['Disagreeing']].iloc[0:49]
mu_scheduling = country_df[['Scheduling']].iloc[0:49]
#mu_persuading = country_df[['Persuading']].iloc[0:50]

y = np.random.normal(mu_leading.loc['USA'], 2.5)
print(y)
print(sector_probs)
sigma = 2.5
country= 'USA'
x = float(mu_scheduling.loc[country])
print(x)

#%% Company agent


class CompanyAgent(mesa.Agent):
    """An agent representing a company that joins SBTi."""

    def __init__(self, unique_id, model, country_probs, sector_probs, mu_communicating, 
                 mu_evaluating, mu_leading, mu_deciding, mu_trusting, mu_disagreeing, 
                 mu_scheduling, sigma):
        super().__init__(unique_id, model)
        
        # PARAMETERS
        
        self.country = self.choose_country(country_probs)
        self.sector = self.choose_sector(sector_probs)
        #self.emissions = None
        self.internal_target = np.random.rand()
        self.shareholdernumber = 1
        
        # Culture dimension values assigned
        self.communicating =    np.clip(np.random.normal(mu_communicating.loc[self.country],sigma),0,100)
        self.evaluating =       np.clip(np.random.normal(mu_evaluating.loc[self.country],sigma),0,100)
        self.leading =          np.clip(np.random.normal(mu_leading.loc[self.country],sigma),0,100)
        self.deciding =         np.clip(np.random.normal(mu_deciding.loc[self.country],sigma),0,100)
        self.trusting =         np.clip(np.random.normal(mu_trusting.loc[self.country],sigma),0,100)
        self.disagreeing =      np.clip(np.random.normal(mu_disagreeing.loc[self.country],sigma),0,100)
        self.scheduling =       np.clip(np.random.normal(mu_scheduling.loc[self.country],sigma),0,100)


        # VARIABLES
        
        self.list_connections = None
        
        #processes
        self.social_pressure = None
        self.motivation = None
        self.target_progress = 0
        
        #states of agent
        self.is_aware = True
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
    
    def commit_to_sbti(self):
        #social pressure
        
        if self in self.model.G:
            neighbors = self.model.G.neighbors(self)
            neighbors_trusting_sum = sum([neighbor.trusting for neighbor in neighbors])
        else:
            neighbors_trusting_sum = 0


        self.social_pressure = self.leading + self.disagreeing + neighbors_trusting_sum
        random_number = np.random.rand()*200

        if self.social_pressure > random_number:
            
            sbti_attributes = self.model.information + self.model.communication + self.model.monitoring + self.model.benefits
            self.motivation  = self.riskawareness * self.disagreeing + self.reputation * self.trusting + self.leadership * self.leading
            
            if sbti_attributes + self.motivation > np.random.rand()*100:
                self.is_committed = True
            
        else:
            self.is_committed = False






    
    
    def step(self):
        num_connections = 0
        if self.model.G.has_node(self):
            num_connections = len(list(self.model.G.neighbors(self)))
        
        
        if not self.is_aware:
            # check if company becomes aware of SBTi
            pass  # code to determine if company becomes aware of SBTi
        elif self.is_aware and not self.is_committed:
            # check if company commits to SBTi
            # code to determine if company commits to SBTi
            self.commit_to_sbti()

            
            

        elif self.is_aware and self.is_committed and not self.has_target:
            self.has_target= True
            
            # check if company sets target with SBTi
            pass  # code to determine if company sets target with SBTi
        else:
            # company has awareness, commitment, and target set with SBTi
            pass  # code for companies with awareness, commitment, and target set]
            
        agent_data.append({
            'Agent ID': self.unique_id,
            'Country': self.country,
            'Sector': self.sector,
            'Internal Target': self.internal_target,
            'Num Connections': num_connections,
            'Communicating':self.communicating,
            'Evaluating': self.evaluating, 
            'Leading': self.leading,
            'Decidng': self.deciding,
            'Trusting': self.trusting,
            'Disagreeing': self.disagreeing,
            'Scheduling': self.scheduling,
            'Social Pressure':self.social_pressure,
            'Corporate motivation': self.motivation,
            'Committed': self.is_committed,
            'Set Target':self.has_target})



#%% SBTi agent

class SBTiModel(mesa.Model):

    def __init__(self, num_companies, country_probs, sector_probs, mu_communicating, 
                 mu_evaluating, mu_leading, mu_deciding, mu_trusting, mu_disagreeing, 
                 mu_scheduling, sigma, seed=None):
        super().__init__()
        self.num_companies = num_companies
        
        
        self.information = np.random.rand()
        self.communication = np.random.rand()
        self.monitoring = np.random.rand()
        self.benefits = np.random.rand()
        
        
        self.schedule = RandomActivation(self)
        self.grid = mesa.space.MultiGrid(20, 20, True)
        
        # define weights for connectivity matrix
        self.alpha = 0.7
        self.beta = 0.3
        
        self.agents = []
        

        
        # add nodes for companies to network
        for i in range(self.num_companies):
             a = CompanyAgent(i, self, country_probs, sector_probs,mu_communicating, 
                          mu_evaluating, mu_leading, mu_deciding, mu_trusting, mu_disagreeing, 
                          mu_scheduling, sigma)
             self.schedule.add(a)
             self.agents.append(a)
             
             # Add the agent to a random grid cell
             x = self.random.randrange(self.grid.width)
             y = self.random.randrange(self.grid.height)
             self.grid.place_agent(a, (x, y))
            
            
        
        # create initial network based on connectivity matrix M
        self.M = self.generateM()
        self.G = self.generateNetwork()
        
        #self.print_agent_connections()
        
        
        # Initialize the NetworkModule to visualize agents and connections
        #self.network =  NetworkModule(self.G, canvas_height=500, canvas_width=800)

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
        return M
    
    def generateNetwork(self):
        # create network based on connectivity matrix M
        G = nx.DiGraph()
        
        for i in range(self.num_companies):
            for j in range(self.num_companies):
                if i != j and random.random() < self.M[i][j]:
                    G.add_edge(self.agents[i], self.agents[j])
        return G
    
    
    #def print_agent_connections(self):
    #    for agent in self.agents:
    #        if self.G.has_node(agent):
    #            num_connections = len(list(self.G.neighbors(agent)))
    #        else:
    #            num_connections = 0
    #        print(f"Agent {agent.unique_id} has {num_connections} connections.")
    
            
    def visualize_network(self):
        # Create a complete graph with all agents as nodes
        all_agents = self.agents
        complete_graph = nx.DiGraph()
        complete_graph.add_nodes_from(all_agents)
        
        # Add the edges based on the actual connections in your model
        complete_graph.add_edges_from(self.G.edges())
        
        pos = nx.spring_layout(complete_graph)
        
        # Create a dictionary mapping agents to their unique_id
        labels = {agent: agent.unique_id for agent in all_agents}
        
        nx.draw(complete_graph, pos, labels=labels, node_color="red", with_labels=True, node_size=200)
        nx.draw_networkx_edge_labels(complete_graph, pos)
        plt.show()
            
    
    def step(self):
        self.schedule.step()

#%%


num_companies = 30
agent_data = []

model = SBTiModel(num_companies, country_probs, sector_probs, mu_communicating, 
             mu_evaluating, mu_leading, mu_deciding, mu_trusting, mu_disagreeing, 
             mu_scheduling, sigma, seed = 2)


for i in range(2):
    model.step()


#model.step()
#model.visualize_network()

edges = model.G.edges(data=True)

connection_matrix = np.zeros((num_companies, num_companies), dtype=bool)

for edge in edges:
    source_agent = edge[0]
    target_agent = edge[1]
    
    source_id = source_agent.unique_id
    target_id = target_agent.unique_id
    connection_matrix[source_id, target_id] = True
    
    #print(f"Agent {source_agent.unique_id} is connected to Agent {target_agent.unique_id}")

df = pd.DataFrame(agent_data)
print(df)


#%%

country_name = 'Ireland'
df1 = df[df.Country == 'Ireland']
print(df1)



# def agent_portrayal(agent):
#     portrayal = {
#         "Shape": "circle",
#         "Filled": "true",
#         "Layer": 0,
#         "Color": "red",
#         "r": 0.5,
#     }
#     return portrayal


# model_params = {"num_companies": num_companies, "country_probs":country_probs, "sector_probs":sector_probs, "seed":123}

# grid = mesa.visualization.CanvasGrid(agent_portrayal, 20, 20, 500, 500)

# server = mesa.visualization.ModularServer(SBTiModel, [grid], "SBTi Model", model_params)

# server.port = 8536  # The default
# server.launch()















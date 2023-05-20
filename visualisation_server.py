#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:00:08 2023

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


from main import SBTiModel, CompanyAgent

#%% Functions


# Specify the path of the Excel file and the name of the worksheet
excel_file_path = 'CHISdata.xlsx'


# Read the data from the worksheet into a dataframe
country_df = pd.read_excel(excel_file_path, sheet_name='Countries', header=0, index_col='Country')
sector_df = pd.read_excel(excel_file_path, sheet_name='Sectors', header=0, index_col='Sector')



country_probs = country_df[['Percentage']].iloc[0:56]
sector_probs = sector_df[['Percentage']].iloc[0:13]




#%%
num_companies = 43


def agent_portrayal(agent):
    num_connections = len(list(agent.model.G.neighbors(agent)))
    
    if num_connections <= 2:
        color = "red"
    elif num_connections <= 4:
        color = "blue"
    else:
        color = "green"

    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "Layer": 0,
        "Color": color,
        "r": 0.5,
    }

    return portrayal


# CHANGED: Subclass CanvasGrid and add edges rendering
class SBTiCanvasGrid(mesa.visualization.CanvasGrid):
    def __init__(self, portrayal_method, grid_width, grid_height, canvas_width, canvas_height):
        super().__init__(portrayal_method, grid_width, grid_height, canvas_width, canvas_height)

    def render(self, model):
        grid_state = super().render(model)
        for agent_portrayal in grid_state["AgentStates"]:
            if "Edges" in agent_portrayal:
                agent_portrayal["Shape"] = "line"
                agent_portrayal["Color"] = "black"
                agent_portrayal["Layer"] = 1
                agent_portrayal["Width"] = 2
                for target_agent_id in agent_portrayal["Edges"]:
                    target_agent = model.agents[target_agent_id]
                    target_x, target_y = target_agent.pos
                    agent_portrayal["x2"] = target_x
                    agent_portrayal["y2"] = target_y
        return grid_state


model_params = {"num_companies": num_companies, "country_probs":country_probs, "sector_probs":sector_probs, "seed":123}

# CHANGED: Use the new SBTiCanvasGrid class
grid = SBTiCanvasGrid(agent_portrayal, 20, 20, 500, 500)

server = mesa.visualization.ModularServer(SBTiModel, [grid], "SBTi Model", model_params)

server.port = 8555  # The default
server.launch()





# # The network_portrayal function now takes an instance of the SBTiModel class
# def network_portrayal(model):
#     G = model_instance.G
#     portrayal = dict()
#     portrayal['nodes'] = [{'size': 10,
#                            'color': 'blue',
#                            'tooltip': f"Agent {agent.unique_id}",
#                            'x': np.random.uniform(-1, 1),
#                            'y': np.random.uniform(-1, 1)}
#                           for agent in G.nodes]
    
#     #portrayal['edges'] = [{'source': edge[0].unique_id,
#     #                       'target': edge[1].unique_id,
#     #                       'color': 'black',
#     #                       'width': 1}
#     #                      for edge in G.edges]

#     return portrayal

# # Run the model with visualization
# num_companies = 43  # Specify the number of companies

# # Global variable to store the SBTiModel instance
# model_instance = None

# def create_model():
#     global model_instance
#     model_instance = SBTiModel(num_companies, country_probs, sector_probs, seed=123)
#     matrix = model_instance.M
#     print(matrix)
#     return model_instance



# # The NetworkModule now takes the network_portrayal function as an argument
# network = NetworkModule(network_portrayal, canvas_height=500, canvas_width=800)

# server = ModularServer(create_model, [network], "SBTi Model")
# matrix = model_instance.M
# server.port = 8529  # Set the port number
# server.launch()

# matrix = model_instance.M





# def network_portrayal(model):
#     G = model.G
#     portrayal = dict()
#     portrayal['nodes'] = [{'size': 10,
#                            'color': 'blue',
#                            'tooltip': f"Agent {agent.unique_id}",
#                            'x': np.random.uniform(-1, 1),
#                            'y': np.random.uniform(-1, 1)}
#                           for agent in G.nodes]
    
#     portrayal['edges'] = [{'source': edge[0].unique_id,
#                            'target': edge[1].unique_id,
#                            'color': 'black',
#                            'width': 1}
#                           for edge in G.edges]

#     return portrayal



# # Run the model with visualization
# num_companies = 43  # Specify the number of companies
# network = NetworkModule(network_portrayal, canvas_height=500, canvas_width=800)

# def create_model():
#      model= SBTiModel(num_companies, country_probs, sector_probs, seed=123)
#      matrix = model.M
#      print(matrix)
#      return model

# server = ModularServer(create_model, [network], "SBTi Model")
# server.port = 8525  # Set the port number
#server.launch()

# # Run the model with visualization
# num_companies = 10  # Specify the number of companies
# model_instance = None

# # Define a function to create the model with specified parameters
# def create_model():
#     global model_instance
#     model_instance = SBTiModel(num_companies, country_probs, sector_probs, seed=123)
#     return model_instance

# # Define the visualization elements for the server
# elements = [create_model().network]
# server = ModularServer(create_model, elements, "SBTi Model")
# server.port = 8521  # Set the port number
# server.launch()

# # Access the connection matrix M
# print("Connection matrix M:")
# print(model_instance.M)



# # Run the model with visualization
# num_companies = 100  # Specify the number of companies
# model = SBTiModel(num_companies, country_probs, sector_probs, seed=123)
# # Define the visualization elements for the server
# elements = [model.network]
# server = ModularServer(model, [model.network], "SBTi Model")
# server.port = 8521  # Set the port number
# server.launch()


# model = SBTiModel(5, country_probs, sector_probs, seed = 2)
# #M = model.generateM()
# #agents = model.schedule

# # for i in range(10):
# #     # Generate the new network
# #     model.step()
# #     new_pos = nx.spring_layout(model.G)

# #     # Plot the new network
# #     fig, ax = plt.subplots(figsize=(10, 10))

# #     nx.draw_networkx_nodes(model.G, pos=new_pos, node_size=50)
# #     nx.draw_networkx_edges(model.G, pos=new_pos)
# #     ax.set_title("Network at Step {i}")
# #    plt.show()  # display the plot in a separate window



# # plot the initial network
# fig, ax = plt.subplots(figsize=(10, 10))
# pos = nx.spring_layout(model.G)
# nx.draw(model.G, pos=pos, with_labels=True, node_size=300)
# ax.set_title("Initial Network")



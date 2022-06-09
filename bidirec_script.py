# script for bidirectional motif counting in temporal networks
# Clark Mollencop Summer 2022

import networkx as nx
import pandas as pd
import numpy as np
import math
import argparse

# function to read the given file with a given separator
def read_file(filename, separator, source_column_index, destination_column_index, timestamp_column_index):
    return pd.read_csv(filename, sep=separator, usecols=[source_column_index, destination_column_index, timestamp_column_index], names=['source', 'target', 'timestamp'], header=0)

# function to duplicate edges for counting bidirectional motifs
# it duplicates edges until and including period + delta 
# returns the graph with the duplicated edges
def dup_edges(graph_df, delta, period=12):
    dup_in_delta = (graph_df.loc[graph_df['timestamp'] <= delta])
    return pd.concat([graph_df, 
                      pd.concat([dup_in_delta.iloc[:,[0, 1]], 
                                 dup_in_delta.iloc[:,[2]]+period], axis=1)], 
                                 ignore_index=True)

# function that sets up the data structure that we use to store
# edge information after we count the bidirectional motifs

# create and fill the data structure for post processing
# TODO use pandas df DONE except for the groupby part for plotting
# so ill leave the comment as is so i know how to plot it
# 3 columns t1, t2, counts
# and then groupby('t1') to plot the graph
# then use itertools.combinations -- 
# t1 is all of the possible timestamps in the network
# change what you had to a dataframe
# want t1 to only have the originals
# and t2 to go all the way through the duplicates
# also want to structure to go the duplicates for t2
# length of thing = unique_len*delta + (unique_len*(unique_len+1))/2 (from math on my desk whiteboard if asked)
def setup_data_struc(graph, graph_dup, delta):
    struc = {'t1': [], 't2': [], 'count': []}
    # number of unique timestamps in the original graph
    unique = pd.unique(graph['timestamp'])
    # number of unique timestamps in the graph with extra timestamps
    unique_dup = pd.unique(graph_dup['timestamp'])
    unique_dup_len = len(unique_dup)
    unique = sorted(unique)
    unique_dup = sorted(unique_dup)
    unique_len = len(unique)
    # length of thing = unique_len*delta + (unique_len*(unique_len+1))/2
    num_pairs = int(unique_len*delta + (unique_len*(unique_len+1))/2)

    for i in range(num_pairs):
        struc['count'].append(0)
    
    for i in range(1, (unique_len+1)):
        j = i
        while j < (unique_dup_len+1):
            struc['t1'].append(i)
            struc['t2'].append(unique_dup[j-1])
            j += 1
    
    # convert this dict to a pandas dataframe for easy plotting
    struc_df = pd.DataFrame.from_dict(struc)
    return struc_df

# convert the graph to a multidigraph in networkx from pandas dataframe
# returns the converted networkx multidigraph
def convert_nx(df):
    G = nx.from_pandas_edgelist(df, 'source', 'target', edge_attr='timestamp', create_using=nx.MultiDiGraph)
    G = nx.convert_node_labels_to_integers(G, ordering='sorted', label_attribute = 'old_id')
    return G

def bidirecCountAndStore(graph, delta, period, struc_df):
    """
    graph: a Time-Varying directed network.
    delta: time-interval within which bidirectional edges must be counted.
    for each edge with timestamp t, computes the number of reverse edges within t + delta and stores this in
    its edge attribute.
    Returns the graph with the additional edge attribute
    """
    
    edges = sorted(graph.edges(data=True), key=lambda t: t[2].get('timestamp', 1))
    
    # select edges that are in original period to avoid redundancy
    # ie they have timestamp <= period
    # with conversion (timestamp -= 1) from earlier, use < not <=
    edges_original = [(u,v) for u,v,e in graph.edges(data=True) if e['timestamp'] < period] 
    bidirec_counts = []
    for i in range(len(edges_original)):
        count = 0
        j = i# + 1
        while j < len(edges):
            if edges[j][2]['timestamp'] > edges[i][2]['timestamp'] + delta:
                break
            elif edges[j][0] == edges[i][1] and edges[j][1] == edges[i][0]:
                count += 1
                t1 = edges[i][2]['timestamp']
                t2 = edges[j][2]['timestamp']
                print(f't1: {t1}, t2: {t2}')
                # TODO be able to do something like struc_df[t1][t2]count+=1
                # there has to be some algebraic relationship between t1, t2,
                # and the pandas df index. I just need to figure it out, maybe 
                # I'll ask abhijin to help me out
            j += 1
        edges[i][2]['delta'][str(delta)] = count
        bidirec_counts.append(count)
    return edges

def main():
    # parge command line arguments
    parser = argparse.ArgumentParser(description='Get bidirectional motif counts in a temporal network.')
    parser.add_argument('source_file', help='Your source graph to count the bidirectional motifs on.')
    # TODO add args for source column name, destination column name, timestamp column name
    parser.add_argument('source_column_index', type=int, help='column in file that contains source node locations')
    parser.add_argument('destination_column_index', type=int, help='column in file that contains destination node locations')
    parser.add_argument('timestamp_column_index', type=int, help='column in file that contains timestamp locations for edges')
    parser.add_argument('separator', type=str, help='The separator in the file containing the graph.')
    parser.add_argument('delta', type=int, help='The delta to use when counting.')
    parser.add_argument('period', type=int, help='The period within which the graph repeats. Eg 12 for a graph that repeats yearly with monthly intervals.')
    args = parser.parse_args()
    
    # read in the file and duplicate its edges for the appropiate delta, period
    df = read_file(args.source_file, args.separator, args.source_column_index, args.destination_column_index, args.timestamp_column_index)
    df_dup = dup_edges(df, delta=args.delta, period=args.period)
    print(df_dup)

    # set up the data structure to store stuff
    data_struc = setup_data_struc(graph=df, graph_dup=df_dup, delta=args.delta)
    print(data_struc)

    # convert with nx stuff
    G = convert_nx(df_dup)
    for e in G.edges(data=True):
        e[2]['delta'] = {}
    list(G.edges(data=True))
    # count the bidirectional edges
    G.edges(data=True)
    edges = bidirecCountAndStore(G, delta = 4, period = 12, struc_df=data_struc)
    bidirecCountAndStore(G, delta = 2, period = 12, struc_df=data_struc)
    G = nx.relabel_nodes(G, nx.get_node_attributes(G, 'old_id'), copy=True)
    print(G.edges(data=True))
    

if __name__ == '__main__':
    main()

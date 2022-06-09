# script for bidirectional motif counting in temporal networks
# Clark Mollencop Summer 2022

import networkx as nx
import pandas as pd
import numpy as np
import math
import argparse

# function to read the given file with a given separator
def read_file(filename, separator):
    return pd.read_csv(filename, sep=separator, names=['source', 'target', 'timestamp'], header=0)

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
def setup_data_struc(graph):
    struc = {'t1': [], 't2': [], 'count': []}
    unique = len(pd.unique(graph['timestamp']))
    num_pairs = math.comb(unique, 2) + unique
    for i in range(num_pairs):
        struc['count'].append(0)
    for i in range(1, (unique+1)):
        j = i
        while j < (unique+1):
            struc['t1'].append(i)
            struc['t2'].append(j)
            j += 1
    return struc

def bidirecCountAndStore(graph, delta):
    """
    graph: a Time-Varying directed network.
    delta: time-interval within which bidirectional edges must be counted.
    for each edge with timestamp t, computes the number of reverse edges within t + delta and stores this in
    its edge attribute.
    Returns the graph with the additional edge attribute
    """
    
    edges = sorted(graph.edges(data=True), key=lambda t: t[2].get('month', 1))
    bidirec_counts = []
    for i in range(len(edges)):
        count = 0
        j = i# + 1
        while j < len(edges):
            if edges[j][2]['month'] > edges[i][2]['month'] + delta:
                break
            elif edges[j][0] == edges[i][1] and edges[j][1] == edges[i][0]:
                count += 1
            j += 1
        edges[i][2]['delta'][str(delta)] = count
        bidirec_counts.append(count)
    return edges

def main():
    # parge command line arguments
    parser = argparse.ArgumentParser(description='Get bidirectional motif counts in a temporal network.')
    parser.add_argument('source_file', help='Your source graph to count the bidirectional motifs on.')
    parser.add_argument('separator', type=str, help='The separator in the file containing the graph.')
    parser.add_argument('delta', type=int, help='The delta to use when counting.')
    parser.add_argument('period', type=int, help='The period within which the graph repeats. Eg 12 for a graph that repeats yearly with monthly intervals.')
    args = parser.parse_args()
    
    # read in the file and duplicate its edges for the appropiate delta, period
    df = read_file(args.source_file, args.separator)
    df_dup = dup_edges(df, delta=args.delta, period=args.period)
    print(df_dup)

    # set up the data structure to store stuff
    data_struc = setup_data_struc(df)
    print(data_struc)

    

if __name__ == '__main__':
    main()
# script for bidirectional motif counting in temporal networks
# Clark Mollencop Summer 2022

import networkx as nx
import pandas as pd
import numpy as np
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
# t1 is all of the possible timestamps in the network
# want t1 to only have the originals
# and t2 to go all the way through the duplicates
# also want to structure to go the duplicates for t2
def setup_data_struc(delta, period):
    struc = {'t1': [], 't2': [], 'count': []}
    
    # all possible timestamp combinations
    num_pairs = period*delta + (period * (period+1))//2

    l = []
    for i in range(1, period + delta + 1):
        l.append(i)
    
    for i in range(num_pairs):
        struc['count'].append(0)
    
    for i in range(1, (period + 1)):
        j = i
        while j < (period + delta + 1):
            struc['t1'].append(i)
            struc['t2'].append(l[j-1])
            j += 1
    
    # convert this dict to a pandas dataframe for easy plotting
    struc_df = pd.DataFrame.from_dict(struc)
    struc_df = struc_df.set_index(['t1', 't2'])
    struc_df['delta'] = delta
    # struc_df.squeeze()
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
    period: period within which the graph repeats 
    struc_df: the dataframe within which we hold the timestamps for plotting
    Returns the graph with the additional edge attribute
    """
    
    edges = sorted(graph.edges(data=True), key=lambda t: t[2].get('timestamp', 1))
    
    # select edges that are in original period to avoid redundancy
    # ie they have timestamp <= period
    # with conversion (timestamp -= 1) from earlier, use < not <=
    # actually I reverted this to fix an off-by-one error, caused by me trying to be too clever
    edges_original = [(u,v) for u,v,e in graph.edges(data=True) if e['timestamp'] <= period] 
    bidirec_counts = []
    for i in range(len(edges_original)):
        count = 0
        j = i + 1
        while j < len(edges):
            if edges[j][2]['timestamp'] > edges[i][2]['timestamp'] + delta:
                break
            elif edges[j][0] == edges[i][1] and edges[j][1] == edges[i][0]:
                count += 1
                t1 = edges[i][2]['timestamp']
                t2 = edges[j][2]['timestamp']
                # update data structure with the counts
                struc_df.loc[(t1, t2)]['count'] += 1
            j += 1
        edges[i][2]['delta'][str(delta)] = count
        bidirec_counts.append(count)
    return edges

# thing to do all of the stuff in main so I can use this script as a module
# all of the arguments are the same as those in main
# sf: source file 
# sci: source column index, the index in the source file where the edge source nodes are listed
# dci: destination column index, the index in the source file where the edge destination nodes are listed
# tci: timestamp column index, the index in the source file where the edge timestamps are listed
# sp: separator in the source file
# d: delta, the time window within which to measure bidirectional motifs
# pd: period, the period within which the graph repeats. eg 12 for a 1-year period with monthly timestamps
# of: output file, the file to which you want to output the results of this count
def doStuff(sf, sci, dci, tci, sp, d, pd, of):
    df = read_file(sf, sp, sci, dci, tci)
    df_dup = dup_edges(df, delta=d, period=pd)

    data_struc = setup_data_struc(delta=d, period=pd)

    G = convert_nx(df_dup)
    for e in G.edges(data=True):
        e[2]['delta'] = {}
    list(G.edges(data=True))

    G.edges(data=True)
    edges = bidirecCountAndStore(G, delta = d, period = pd, struc_df=data_struc)
    
    G = nx.relabel_nodes(G, nx.get_node_attributes(G, 'old_id'), copy=True)
    
    data_struc.to_csv(of)


def main():
    # parge command line arguments
    parser = argparse.ArgumentParser(description='Get bidirectional motif counts in a temporal network.')
    parser.add_argument('source_file', help='Your source graph to count the bidirectional motifs on.')
    parser.add_argument('source_column_index', type=int, help='column in file that contains source node locations')
    parser.add_argument('destination_column_index', type=int, help='column in file that contains destination node locations')
    parser.add_argument('timestamp_column_index', type=int, help='column in file that contains timestamp locations for edges')
    parser.add_argument('separator', type=str, help='The separator in the file containing the graph.')
    parser.add_argument('delta', type=int, help='The delta to use when counting.')
    parser.add_argument('period', type=int, help='The period within which the graph repeats. Eg 12 for a graph that repeats yearly with monthly intervals.')
    parser.add_argument('output_file', type=str, help='File to output the results of this script.')
    args = parser.parse_args()
    
    # read in the file and duplicate its edges for the appropiate delta, period
    df = read_file(args.source_file, args.separator, args.source_column_index, args.destination_column_index, args.timestamp_column_index)
    df_dup = dup_edges(df, delta=args.delta, period=args.period)
    # print(df_dup)

    # set up the data structure to store stuff
    data_struc = setup_data_struc(delta=args.delta, period=args.period)
    # data_struc = setup_data_struc(graph=df, graph_dup=df_dup, delta=3, period=5)
    # print(data_struc.to_string())

    # convert with nx stuff
    G = convert_nx(df_dup)
    for e in G.edges(data=True):
        e[2]['delta'] = {}
    list(G.edges(data=True))
    # count the bidirectional edges
    G.edges(data=True)
    edges = bidirecCountAndStore(G, delta = args.delta, period = args.period, struc_df=data_struc)
    
    G = nx.relabel_nodes(G, nx.get_node_attributes(G, 'old_id'), copy=True)
    # print(G.edges(data=True))
    # print(data_struc.to_string())
    data_struc.to_csv(args.output_file)

if __name__ == '__main__':
    main()

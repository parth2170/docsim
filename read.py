import sys
import numpy as np 
import pandas as pd
import networkx as nx
import datetime 
from node2vec import Node2Vec 


def read_data(path):
	t1 = datetime.datetime.now()
	data = {}
	with open(path, 'r') as file:
		for line in file:
			temp = line.split("-->")
			case = temp[0]
			temp[1].replace("$$ ", "$$$")
			temp = temp[1][:-1].split("$$$")
			acts = []
			for i in range(len(temp)):
				t = temp[i].split(" or ")
				if(len(t) > 1):
					prefix = t[0].split("_")[0]
					for j in range(1, len(t)):
						t[j] = (prefix+t[j]).strip()
					acts.extend(t)
				else:
					acts.append(t[0].strip())
			data[case] = list(set(acts))
	t2 = datetime.datetime.now()
	print("File Read, Time taken = {} microseconds".format((t2-t1).microseconds))
	return data

def graph(data):
	#Get node list and edge list from data
	t1 = datetime.datetime.now()
	case_nodes = []
	act_nodes = []
	edges = []
	for i in data.items():
		case_nodes.append(i[0])
		act_nodes.extend(i[1])
		for j in i[1]:
			edges.append((i[0], j))
	act_nodes = list(set(act_nodes))
	print("Number of cases = {}".format(len(case_nodes)))
	print("Number of acts = {}".format(len(act_nodes)))
	print("Number of edges = {}".format(len(edges)))
	t2 = datetime.datetime.now()
	print("Time taken to get grapgh data= {} microseconds".format((t2-t1).microseconds))
	return case_nodes, act_nodes, edges

def node2vec(case_nodes, act_nodes, edges):
	#Creating graph and running node2vec
	G = nx.Graph()
	G.add_nodes_from(case_nodes)
	G.add_nodes_from(act_nodes)
	G.add_edges_from(edges)
	node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4) 
	model = node2vec.fit(window=5, min_count=1, batch_words=5)
	model.wv.save_word2vec_format("/saved/sample_node2vec_embeddings")
	model.save("/saved/sample_node2vec_model")



def main():

	if len(sys.argv) < 3:
		print("Enter the training file path as first arguement and the testing file path as the second arguement")
		print("for example if the files are in the same directory as this script enter \npython read.py docs_secs.txt docs_sec_test.txt")
		return 

	train = sys.argv[1]
	test = sys.argv[2]
	train_data = read_data(train)
	test_data = read_data(test)
	#Merging the two dictionaries to build graph
	data = {**train_data, **test_data}
	case_nodes, act_nodes, edges = graph(data = data)

if __name__ == '__main__':
	main()

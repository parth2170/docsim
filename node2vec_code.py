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
			if case == '1995_E_8.txt':
				continue
			temp[1] = temp[1].replace("$$ ", "$$$")
			temp = temp[1][:-1].split("$$$")
			acts = []
			for i in range(len(temp)):
				t = temp[i].split(" or ")
				if(len(t) > 1):
					prefix = t[0].split("_")[0] + "-"
					prefix = prefix.strip('$')
					for j in range(1, len(t)):
						t[j] = (prefix+t[j]).strip()
					acts.extend(t)
				else:
					t[0] = t[0].strip()
					t[0] = t[0].strip("$")
					acts.append(t[0].strip())
			data[case] = list(set(acts))
	t2 = datetime.datetime.now()
	print("File Read, Time taken = {} microseconds".format((t2-t1).microseconds))
	return data

def graph(data, gpath):
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
	G = nx.Graph()
	G.add_nodes_from(case_nodes)
	G.add_nodes_from(act_nodes)
	G.add_edges_from(edges)
	nx.write_gpickle(G, gpath)
	print("Graph saved at "+gpath)
	t2 = datetime.datetime.now()
	print("Time taken to get grapgh data= {} microseconds".format((t2-t1).microseconds))
	return case_nodes, act_nodes, edges

def node2vec(gpath):
	embout = "saved/node2vec_embeddings"
	modelout = "saved/node2vec_model"
	G = nx.read_gpickle(gpath)
	node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4) 
	model = node2vec.fit(window=5, min_count=1, batch_words=5)
	model.wv.save_word2vec_format(embout)
	model.save(modelout)
	print("Model saved at "+modelout)
	print("Embeddings saved at "+embout)
	return model

def distace(model, distpath, distout):
	fo = open(distout, 'w')
	with open(distpath, 'r') as file:
		for line in file:
			temp = line.split()
			case1, case2 = temp[0]+".txt", temp[1]+".txt"
			try:
				sim = model.wv.similarity(case1, case2)
				outline = temp[0]+" "+temp[1]+" "+str(sim)
				fo.write(outline+"\n")
			except KeyError as error:
				print(error)
	fo.close()
	print("Similarity scores saved at "+distout)


def main():
	train = "docs_secs.txt"
	test = "docs_sec_test.txt"
	distpath = "test_scores.txt"
	distout = "saved/node2vec_scores.txt"
	gpath = "saved/node2vec_graph.gpickle"

	print("Please specify all the parameters and paths in the script itself")
	print("Enter 1 to read data and generate graph")
	print("Enter 2 to run node2vec on the generated graph and find similarity scores")
	print("Enter 3 to do all")
	model = None 
	task = int(input("Enter : "))
	if task == 1 or task == 3:
		train_data = read_data(train)
		test_data = read_data(test)
		data = {**train_data, **test_data}
		case_nodes, act_nodes, edges = graph(data = data, gpath = gpath)
	if task == 2 or task == 3:
		model = node2vec(gpath)
		distace(model = model, distpath = distpath, distout = distout)
	else:
		print("Invalid Choice")

if __name__ == '__main__':
	main()

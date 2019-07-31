import numpy as np 
import pickle
import random
from tqdm import tqdm
import psutil
import os
import datetime
import networkx as nx
from node2vec import Node2Vec 


def reverse_dict(D):
	rD = {}
	for i in D:
		for j in D[i]:
			try:
				rD[j].append(i)
			except KeyError:
				rD[j] = []
				rD[j].append(i)
	return rD

def readsmall():
	case_sec = {}
	with open('doc-sec.txt') as file:
		for line in file:
			case, secs = line.split('-->')
			secs = secs.split('$$$')
			secs[-1] = secs[-1][:-1]
			for i in range(len(secs)):
				secs[i] = secs[i].replace(' ', '*')
			try:
				case_sec[case].extend(secs)
			except KeyError:
				case_sec[case] = []
				case_sec[case].extend(secs)
	sec_case = reverse_dict(case_sec)
	return case_sec, sec_case

def readbig():
	case_sec = {}
	with open('train-doc-sec-citations.txt') as file:
		for line in file:
			case, secs = line.split('-->')
			case = case[:-4]
			secs = secs.split('$$$')
			secs[-1] = secs[-1][:-1]
			for i in range(len(secs)):
				secs[i] = secs[i].replace(' ', '*')
			try:
				case_sec[case].extend(secs)
			except KeyError:
				case_sec[case] = []
				case_sec[case].extend(secs)
	#sec_case = reverse_dict(case_sec)
	return case_sec

def readact():
	act_sec = {}
	with open('sec_partof_act.txt') as file:
		for line in file:
			act = line.split('\t')[0]
			secs = line.split('\t')[1:]
			if len(secs) > 1:
				continue
			secs = secs[0][:-1]
			secs = secs.replace(' ', '*')
			act = act.replace(' ', '*')
			try:
				act_sec[act].append(secs)
			except KeyError:
				act_sec[act] = []
				act_sec[act].append(secs)
	sec_act = reverse_dict(act_sec)
	return act_sec, sec_act

def metapaths1(case, case_sec, sec_case, numwalks, walklength):
	#Generating meta-paths for metapath2vec
	outfile = []
	case0 = case
	for j in range(numwalks):
		outline = case0
		for i in range(walklength):
			secs = case_sec[case]
			nums = len(secs)
			sec_id = random.randrange(nums)
			sec = secs[sec_id]
			outline += " " + str(sec)
			cases = sec_case[sec]
			numc = len(cases)
			caseid = random.randrange(numc)
			case = cases[caseid]
			outline += " " + str(case)
		outfile.append(outline + "\n")
	return outfile

def metapaths2(case, case_sec, sec_case, act_sec, sec_act, numwalks, walklength):
	#Generating meta-paths for metapath2vec
	outfile = []
	case0 = case
	for j in range(numwalks):
		outline = case0
		for i in range(walklength):
			secs = case_sec[case]
			nums = len(secs)
			sec_id = random.randrange(nums)
			sec = secs[sec_id]
			outline += " " + str(sec)
			try:
				acts = sec_act[sec]
			except:
				continue
			numa = len(acts)
			act_id = random.randrange(numa)
			act = acts[act_id]
			outline += " " + str(act)
			secs = act_sec[act]
			nums = len(secs)
			sec_id = random.randrange(nums)
			sec = secs[sec_id]
			outline += " " + str(sec)
			try:
				cases = sec_case[sec]
			except:
				continue
			numc = len(cases)
			caseid = random.randrange(numc)
			case = cases[caseid]
			outline += " " + str(case)
		outfile.append(outline + "\n")
	return outfile


def metapath2vec(small, size, types):
	print("Running Metapath2Vec")
	os.chdir('../code_metapath2vec')
	pp = 1
	window = 5
	negative = 3
	outpath = "../metanew/" + small + "_paths" + str(types) + ".txt"
	embout = "../metanew/" + "meta" + small + "_emb" + str(types)
	cmd = "./metapath2vec -train "+outpath+" -output "+embout+" -pp "+str(pp)+" -size "+str(size)+" -window "+str(window)+" -negative "+str(negative)+" -threads 32"
	os.system(cmd)
	print("Embddings saved at "+embout)

def pathmaker(case_sec, sec_case, act_sec, sec_act, small):
	r1, r2 = [], []
	print('Generating paths')
	for case in tqdm(case_sec):
		path1 = metapaths1(case, case_sec, sec_case, numwalks = 20, walklength = 10)
		path2 = metapaths2(case, case_sec, sec_case, act_sec, sec_act, numwalks = 20, walklength = 10)
		r1.append(path1)
		r2.append(path2)
	bp1 = open(small + "_paths1.txt", 'w')
	bp2 = open(small + "_paths2.txt", 'w')
	for paths in r1:
		for path in paths:
			bp1.write(path)
	for paths in r2:
		for path in paths:
			bp2.write(path)
	bp1.close()
	bp2.close()

def node2vec_graph(D):
	t1 = datetime.datetime.now()
	G = nx.Graph(D)
	print('Graph generated')
	#nx.write_gpickle(G, gpath+'network.gpickle')
	#print("Graph saved at "+gpath)
	t2 = datetime.datetime.now()
	print(nx.info(G))
	print("Time taken to get graph data = {} microseconds".format((t2-t1).microseconds))
	return G

def node2vec(G):
	embout = 'node_big_emb'
	node2vec = Node2Vec(G, dimensions=32, walk_length=20, num_walks=10, workers=int(psutil.cpu_count())) 
	model = node2vec.fit(window=5, min_count=1, batch_words=5)
	print('Saving')
	model.wv.save_word2vec_format(embout)
	model.save('node2vec.model')
	return model


def nodesim(model):
	fo = open('nose_sim.txt', 'w')
	text = []
	ann = []
	net = []
	with open('../test_scores.txt', 'r') as file:
		for line in file:
			case1, case2 = line.split()[0], line.split()[1]
			text.append(line.split()[-1])
			ann.append(line.split()[-2])
			try:
				sim = model.wv.similarity(case1, case2)
				net.append(sim)
				outline = case1 + " " + case2 + " " + str(sim)
				fo.write(outline + "\n")
			except KeyError as error:
				print(error)
	ann = ann / np.linalg.norm(ann)
	print('-----Correlations-----')
	print('Text - Annotated = {}'.format(np.corrcoef(text, ann)[0][1]))
	print('Text - Network = {}'.format(np.corrcoef(text, net)[0][1]))
	print('Network - Annotated = {}'.format(np.corrcoef(net, ann)[0][1]))
	fo.close()



def main():
	scase_sec, ssec_case = readsmall()
	bcase_sec = readbig()
	act_sec, sec_act = readact()
	case_sec = {**scase_sec, **bcase_sec}
	sec_case = reverse_dict(case_sec)
	# with open('cs.pickle', 'wb') as file:
	# 	pickle.dump(case_sec, file)
	# print('Running Metapath2Vec')
	# pathmaker(case_sec, sec_case, act_sec, sec_act, small = "big")
	# pathmaker(scase_sec, ssec_case, act_sec, sec_act, small = "small")
	# metapath2vec(small = "big", size = 32, types = 1)
	# metapath2vec(small = "big", size = 64, types = 2)
	print('Running Node2Vec')
	G = node2vec_graph(case_sec)
	model = node2vec(G)

	
if __name__ == '__main__':
	main()
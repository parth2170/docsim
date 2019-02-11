import datetime
import pickle
import pandas as pd 
from tqdm import tqdm
import psutil
import progressbar
import networkx as nx
from node2vec import Node2Vec 
from gensim.models import Word2Vec

def reverse_dict(D):
	rD = {}
	for i in tqdm(D):
		for j in D[i]:
			try:
				rD[j].append(i)
			except KeyError:
				rD[j] = []
				rD[j].append(i)
	return rD

def special(D):
	'''
	clean = {}
	for case in D:
		if len(D[case]) == 1:
			clean[case] = D[case]
		for act in D[case]:
			temp = act.split(',')
			if len(temp) != 2:
				print(act)
		print('')
	'''
	return D

no_act_cases = []

def read_data(path):
	print('Reading Test data')
	test = read_test('parth_kg_embed/test_cases.txt')
	case_act_sec_dict = {}
	act_case_dict = {}
	all_cases = []
	all_acts_sec = []
	no_date_acts_unique = 0
	no_date_acts_multiple = 0
	special_cases = {}
	count = 0
	print('Making Case --> Act Dictionary')
	bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
	with open(path, 'r') as file:
		for line in file:
			bar.update(count)
			count += 1
			temp = line.split(".txt-->")
			if len(temp) == 1:
				temp = line.split(".txt--->")
			case = temp[0]
			case_act_sec_dict[case] = []
			all_cases.append(case)
			if len(temp) == 2 and (temp[1] == '\n' or temp[1] == ''):
				case_act_sec_dict.pop(case, None)
				no_act_cases.append(case)
				continue
			acts = temp[1].split('$$$')
			if acts[-1][-1] == '\n':
				acts[-1] = acts[-1][:-1]
			ycheck = None
			actcheck = [x.split(',') for x in acts]
			#deleter = []
			flag = 0
			for q in range(len(actcheck)):
				if len(actcheck[q]) > 2:
					#When this occurs, there are only special cases in acts
					special_cases[case] = acts
					flag = 1
			if flag:
				continue
			df = pd.DataFrame(actcheck)
			if len(df.columns) == 1:
				case_act_sec_dict[case] = acts
				all_acts_sec.extend(acts)
				continue
			deleter = []
			for index, row in df.iterrows():
				if(row[1] == None):
					ycheck = row[0].split('_')[0]
					years = []
					for indexj, rowj in df.iterrows():
						if ycheck == rowj[0].split('_')[0] and rowj[1] != None:
							years.append(rowj[1].split('_')[0])
					if len(set(years)) == 1:
						if row[0].split('_')[1]:
							row[1] = years[0] + '_' + row[0].split('_')[1]
							row[0] = row[0].split('_')[0]
						no_date_acts_unique += 1
					elif len(set(years)) > 1:
						deleter.append(index)
						no_date_acts_multiple += 1

			acts = []
			deleter = list(set(deleter))
			df = df.drop(df.index[deleter])
			for index, row in df.iterrows():
				if row[1] == None:
					acts.append(row[0])
				else:
					acts.append(row[0] + ',' + row[1])
			case_act_sec_dict[case] = acts
			all_acts_sec.extend(acts)
	print('\nNumber of special cases = {}'.format(len(special_cases)))
	print('Handling Special cases')
	#Handle special cases and append to main dictionaries
	#special_cases = special(special_cases)
	case_act_sec_dict = dict(case_act_sec_dict, **special_cases)
	case_act_sec_dict = dict(case_act_sec_dict, **test)
	print('Making Act --> Case Dictionary')
	act_case_dict = reverse_dict(case_act_sec_dict)
	all_acts_sec = list(set(all_acts_sec))
	print('Total number of acts_sections cited = {}'.format(len(all_acts_sec)))
	print('No act cases = {}'.format(len(no_act_cases)))
	print('no_date_acts_unique = {}'.format(no_date_acts_unique))
	print('no_date_acts_multiple = {}'.format(no_date_acts_multiple))
	print('Saving')
	#Save the dictionaries
	with open('saved/case_act_sec_dict.pickle', 'wb') as file:
		pickle.dump(case_act_sec_dict, file)
	with open('saved/act_case_dict.pickle', 'wb') as file:
		pickle.dump(act_case_dict, file)
	
	return case_act_sec_dict, act_case_dict

def read_test(path):
	t1 = datetime.datetime.now()
	data = {}
	with open(path, 'r') as file:
		for line in file:
			temp = line.split(".txt-->")
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


def node2vec_graph(D, gpath):
	t1 = datetime.datetime.now()
	G = nx.Graph(D)
	print('Graph generated, now saving')
	nx.write_gpickle(G, gpath+'network.gpickle')
	print("Graph saved at "+gpath)
	t2 = datetime.datetime.now()
	print("Time taken to get graph data= {} microseconds".format((t2-t1).microseconds))
	print(nx.info(G))
	return G

def node2vec(G, gpath):
	embout = gpath+'Embeddings128'
	node2vec = Node2Vec(G, dimensions=128, walk_length=30, num_walks=200, workers=int(psutil.cpu_count())) 
	model = node2vec.fit(window=5, min_count=1, batch_words=5)
	print('Saving')
	model.wv.save_word2vec_format(embout)
	model.save(gpath+'node2vec.model128')
	print('Getting Similarity')
	return model

def distace(model, distpath, distout):
	fo = open(distout, 'w')
	with open(distpath, 'r') as file:
		for line in file:
			temp = line.split('-->')
			case1, case2 = temp[0].split('.')[0], temp[1].split('.')[0]
			if case1 in no_act_cases:
				print('{} does not cite any act'.format(case1))
				break
			if case2 in no_act_cases:
				print('{} does not cite any act'.format(case2))
				break
			try:
				sim = model.wv.similarity(case1, case2)
				outline = temp[0]+" "+temp[1]+" "+str(sim)
				fo.write(outline+"\n")
			except KeyError as error:
				print(error)
	fo.close()
	print("Similarity scores saved at "+distout)


if __name__ == '__main__':

	gpath = 'saved/'
	dict_path = 'saved/case_act_sec_dict.pickle'

	print('Enter 1 to Read Data and save dictionaries')
	print('Enter 2 to generate and save node2vec graph')
	print('Enter 3 to run node2vec on generated graph')
	print('Enter 4 to get Embeddings similarity')
	print('Enter 5 to do both 2 and 3')
	print('Enter 6 to do all')
	q = int(input('Enter : '))
	if q == 6:
		c_a, a_c = read_data('parth_kg_embed/doc-sec-cit.txt')
		model = node2vec(node2vec_graph(c_a, gpath), gpath)
		#Negative Samples
		distace(model, 'parth_kg_embed/test/negative.txt', gpath+'node2vec_negsim128.txt')
		#Positive Samples
		distace(model, 'parth_kg_embed/test/positive.txt', gpath+'node2vec_possim128.txt')
	elif q == 5:
		with open(dict_path, 'rb') as file:
			c_a = pickle.load(file)
		model = node2vec(node2vec_graph(c_a, gpath), gpath)
		#Negative Samples
		distace(model, 'parth_kg_embed/test/negative.txt', gpath+'node2vec_negsim128.txt')
		#Positive Samples
		distace(model, 'parth_kg_embed/test/positive.txt', gpath+'node2vec_possim128.txt')
	elif q == 4:
		model = Word2Vec.load(gpath+'node2vec.model')
		#Negative Samples
		distace(model, 'parth_kg_embed/test/negative.txt', gpath+'node2vec_negsim128.txt')
		#Positive Samples
		distace(model, 'parth_kg_embed/test/positive.txt', gpath+'node2vec_possim128.txt')
	elif q == 3:
		G = nx.read_gpickle(gpath+'network.gpickle')
		node2vec(G, gpath)
	elif q == 2:
		with open(dict_path, 'rb') as file:
			c_a = pickle.load(file)
		G = node2vec_graph(c_a, gpath)
	elif q == 1:
		c_a, a_c = read_data('parth_kg_embed/doc-sec-cit.txt')
	else:
		print('Invalid Choice')



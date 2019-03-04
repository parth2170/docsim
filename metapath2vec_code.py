import os
import datetime
import random
import pickle
from progressbar import ProgressBar
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
pbar = ProgressBar()

def metapath_gen(case, act_case_dict, outpath, numwalks, walklength, case_act_dict):
	#Generating meta-paths for metapath2vec
	outfile = []
	case0 = case
	for j in range(numwalks):
		outline = case0
		for i in range(walklength):
			acts = case_act_dict[case]
			numa = len(acts)
			actid = random.randrange(numa)
			act = acts[actid]
			outline += " " + str(act)
			cases = act_case_dict[act]
			numc = len(cases)
			caseid = random.randrange(numc)
			case = cases[caseid]
			outline += " " + str(case)
		outfile.append(outline + "\n")
	return outfile

def metapath2vec(code_dir, outpath, embout):
	print("Running Metapath2Vec")
	os.chdir(code_dir)
	pp = 1
	size = 128
	window = 7
	negative = 5
	outpath = "../"+outpath
	embout = "../"+embout
	cmd = "./metapath2vec -train "+outpath+" -output "+embout+" -pp "+str(pp)+" -size "+str(size)+" -window "+str(window)+" -negative "+str(negative)+" -threads 32"
	os.system(cmd)
	print("Embddings saved at "+embout)

def distance(code_dir, embout):
	os.chdir(code_dir)	
	cmd = "./distance ../"+embout
	os.system(cmd)

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

def make_acts_to_case(act_case_dict):
	act_case = {}
	print('Making act-->case dict')
	for sec in tqdm(act_case_dict):
		cases = act_case_dict[sec]
		act = sec.split('_')[0]
		try:
			act_case[act].extend(cases)
		except:
			act_case[act] = cases
	act_case = act_codes(act_case)
	print('Making case-->act dictionay')
	case_act_dict = reverse_dict(act_case)
	return case_act_dict, act_case

def act_codes(act_case_dict):
	print('Coding Acts')
	j = 0
	codes = {}
	for i in act_case_dict:
		codes[j] = act_case_dict[i]
		j += 1
	return codes

def main():
	outpath = "saved/metapaths.txt"
	embout = "saved/metapath2vec_embeddings"
	metapath2vec_dir = "/Users/deepthought/code/docsim/code_metapath2vec"

	print("Please specify all the parameters and paths in the script itself")
	print("Enter 0 to run on small network only")
	print("Enter 1 to generate metapaths")
	print("Enter 2 to run metapath2vec on generated metapaths")
	print("Enter 3 to run distance on generated embeddings")
	print("Enter 4 to 1 and 2")
	task = int(input("Enter : "))
	if task == 1 or task == 4 or task ==0:
		numwalks = 100
		walklength = 30
		if task == 0:
			with open('saved/smallact_case_dict.pickle', 'rb') as file:
				act_case_dict = pickle.load(file)
			outpath = "saved/small_metapaths.txt"
			embout = "saved/small_metapath2vec_embeddings"
		else:	
			with open('saved/sec_case_dict.pickle', 'rb') as file:
				act_case_sec_dict = pickle.load(file)
		case_act_dict, act_case_dict = make_acts_to_case(act_case_sec_dict)
		num_cores = multiprocessing.cpu_count()
		results = []
		for i in tqdm(case_act_dict):
			results.append(metapath_gen(case = i, act_case_dict = act_case_dict, outpath = outpath, numwalks = numwalks, walklength =  walklength, case_act_dict = case_act_dict))
		print('Saving metapaths as txt')
		with open(outpath, 'w') as file:
			for i in tqdm(results):
				for j in i:
					file.write(j)
	if task == 2 or task == 4 or task == 0:
		metapath2vec(code_dir = metapath2vec_dir, outpath = outpath, embout = embout)
	if task == 3:
		distance(code_dir = metapath2vec_dir, embout = embout)
	else:
		print("Invalid Choice")

if __name__ == '__main__':
	main()

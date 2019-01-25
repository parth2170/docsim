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
			outline += " " + act
			cases = act_case_dict[act]
			numc = len(cases)
			caseid = random.randrange(numc)
			case = cases[caseid]
			outline += " " + case
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

def main():
	outpath = "saved/metapaths.txt"
	embout = "saved/metapath2vec_embeddings"
	metapath2vec_dir = "/Users/deepthought/code/docsim/code_metapath2vec"

	print("Please specify all the parameters and paths in the script itself")
	print("Enter 1 to generate metapaths")
	print("Enter 2 to run metapath2vec on generated metapaths")
	print("Enter 3 to run distance on generated embeddings")
	print("Enter 4 to 1 and 2")
	task = int(input("Enter : "))
	if task == 1 or task == 4:
		numwalks = 200
		walklength = 30
		with open('saved/case_act_sec_dict.pickle', 'rb') as file:
			case_act_dict = pickle.load(file)
		with open('saved/act_case_dict.pickle', 'rb') as file:
			act_case_dict = pickle.load(file)
		num_cores = multiprocessing.cpu_count()
		print('Running on {} cores'.format(num_cores))
		results = Parallel(n_jobs=num_cores)(delayed(metapath_gen)(case = i, act_case_dict = act_case_dict, outpath = outpath, numwalks = numwalks, walklength =  walklength, case_act_dict = case_act_dict) for i in tqdm(case_act_dict))	
	if task == 2 or task == 4:
		metapath2vec(code_dir = metapath2vec_dir, outpath = outpath, embout = embout)
	if task == 3:
		distance(code_dir = metapath2vec_dir, embout = embout)
	else:
		print("Invalid Choice")

if __name__ == '__main__':
	main()

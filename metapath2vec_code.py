import os
import datetime
import random
from progressbar import ProgressBar
pbar = ProgressBar()

def read_data(path):
	#returns case:act dictionary and act:case dictionary
	t1 = datetime.datetime.now()
	case_act_dict = {}
	act_case_dict = {}
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
					prefix = t[0].split("_")[0] + "_"
					prefix = prefix.strip('$')
					for j in range(1, len(t)):
						t[j] = (prefix+t[j]).strip()
						if t[j] not in act_case_dict:
							act_case_dict[t[j].strip()] = []
						act_case_dict[t[j].strip()].append(case.strip())
					if t[0] not in act_case_dict:
						act_case_dict[t[0].strip()] = []
					act_case_dict[t[0].strip()].append(case.strip())
					acts.extend(t)
				else:
					t[0] = t[0].strip()
					t[0] = t[0].strip("$")
					if t[0] not in act_case_dict:
						act_case_dict[t[0].strip()] = []
					act_case_dict[t[0].strip()].append(case.strip())
					acts.append(t[0].strip())
			case_act_dict[case] = list(set(acts))
	t2 = datetime.datetime.now()
	print("File Read, Time taken = {} microseconds".format((t2-t1).microseconds))
	return case_act_dict, act_case_dict


def metapath_gen(case_act_dict, act_case_dict, outpath, numwalks, walklength):
	#Generating meta-paths for metapath2vec
	print("Number of cases = {}".format(len(case_act_dict)))
	print("Number of acts = {}".format(len(act_case_dict)))
	print("Generating Metapaths")
	outfile = open(outpath, 'w')
	for case in pbar(case_act_dict):
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
			outfile.write(outline + "\n")
	outfile.close()
	print("Meta-paths saved at "+outpath)

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
	cmd = "./distance "+embout
	os.system(cmd)

def main():
	train = "docs_secs.txt"
	test = "docs_sec_test.txt"
	outpath = "saved/metapaths.txt"
	embout = "saved/metapath2vec_embeddings.txt"
	metapath2vec_dir = "/Users/deepthought/code/docsim/code_metapath2vec"

	print("Please specify all the parameters and paths in the script itself")
	print("Enter 1 to read data and generate metapaths")
	print("Enter 2 to run metapath2vec on generated metapaths")
	print("Enter 3 to run distance on generated embeddings")
	print("Enter 4 to do all")
	task = int(input("Enter : "))
	if task == 1 or task == 4:
		numwalks = 200
		walklength = 30
		case_act_dict1, act_case_dict1 = read_data(train)
		case_act_dict2, act_case_dict2 = read_data(test)
		case_act_dict = {**case_act_dict1, **case_act_dict2}
		act_case_dict = {**act_case_dict1, **act_case_dict2}
		metapath_gen(case_act_dict = case_act_dict, act_case_dict = act_case_dict, outpath = outpath, numwalks = numwalks, walklength =  walklength)
	if task == 2 or task == 4:
		metapath2vec(code_dir = metapath2vec_dir, outpath = outpath, embout = embout)
	if task == 3 or task == 4:
		distance(code_dir, embout)
	else:
		print("Invalid Choice")

if __name__ == '__main__':
	main()

import csv
from math import exp, log, floor
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

P_THRESH = 0.5
CONF_THRESH = 0
HYPO_DIFF_THRESH = 0.75
HYPO_DIFF_THRESH = 0.5
C_VAL = 0.001
G_VAL = 1

def parse_csv (foldernames, confidence_thresh1=CONF_THRESH):
	gender = {'1':'male','2':'female','3':'other'}
	handedness = {'1':'left','2':'right','3':'mixed','4':'ambidextrous'}
	eyesight = {'1':'perfect','2':'corrected','3':'incorrect'}
	subject_info = dict()
	print 'Selecting subjects with confidence in exp1 no less than', confidence_thresh1
	for foldername in foldernames:
		filename = foldername+'/data.csv'
		with open(filename) as csvfile:
			subjects = csv.DictReader(csvfile)
			for subject in subjects:
				if (subject['endcode'] != '') and (subject['endcode'] not in subject_info) and (int(subject['confidence_1:1'])>=confidence_thresh1): #ensure only completed surveys are taken into account
					subject_info[subject['endcode']] = {'endcode':subject['endcode'],'name':subject['name:1'].lower(),'entrynumber':subject['entryno:1'].lower(),'age':subject['age:1'],'gender':gender[subject['gender:1']],'handedness':handedness[subject['handedness:1']],'eyesight':eyesight[subject['eyesight:1']],'exp1file':(foldername+'/'+subject['category_experiment_1:1']),'exp1confidence':subject['confidence_1:1'],'exp1difficulty':subject['difficulty_1:1'],'exp1fatigue':subject['fatigue_1:1'],'exp2file':(foldername+'/'+subject['category_experiment_2:1']),'exp2confidence':subject['confidence_2:1'],'exp2difficulty':subject['difficulty_2:1'],'exp2fatigue':subject['fatigue_2:1'],'description':subject['test:1'],'email':subject['email:1'],'timestart':subject['TIME_start'],'timeend':subject['TIME_end']}
	return subject_info

def parse_exp (filename):
	with open(filename) as csvfile:
		samples = csv.reader(csvfile, delimiter=' ')
		exp_info = dict()
		l_or_r = {'-300':'l','300':'r'}
		a_or_b = {'0':'X','71':'a','72':'b'}
		r_or_w = {'1':1,'2':0,'3':-1}
		for sample in samples:
			value = sample[1:]
			value[1] = l_or_r[value[1]]
			value[2] = a_or_b[value[2]]
			value[3] = r_or_w[value[3]]
			value[4] = int(value[4])
			if sample[0] in exp_info:
				exp_info[sample[0]]['all'].append(value)
				if value[1] == 'l':
					exp_info[sample[0]]['l'].append(value)
				else:
					exp_info[sample[0]]['r'].append(value)
			else:
				exp_info[sample[0]] = {'all':[value],'l':[],'r':[]}
				if value[1] == 'l':
					exp_info[sample[0]]['l'].append(value)
				else:
					exp_info[sample[0]]['r'].append(value)
	return exp_info

def parse_img(filename):
	with open(filename) as csvfile:
		samples = csv.reader(csvfile, delimiter=',')
		rep_set = []
		for sample in samples:
			rep_set.append([float(i) for i in sample])
		rep_set = np.asarray(rep_set)
		rep_set = (rep_set-rep_set.mean(axis=0))/rep_set.std(axis=0) #zero mean unit variance
		rep_set = rep_set.tolist()
		n = len(rep_set)
		rep_set = dict([('a'+str(i%(n/2)),rep_set[i]) if i<n/2 else ('b'+str(i%(n/2)),rep_set[i]) for i in range(n)])
	return rep_set

def similarity_images(img, imglist, c):
	total_sim = 0.0
	for imgcomp in imglist:
		total_sim += exp(-c*(sum([(img[i]-imgcomp[i])**2 for i in range(len(img))])**0.5))
	return total_sim

def similarity_images(img, imglist, c): #uncomment this function out to use cosine similarity, else default is exponential similarity
	total_sim = 0.0
	for imgcomp in imglist:
		total_sim += abs(np.dot(img,imgcomp)/(np.linalg.norm(img)*np.linalg.norm(imgcomp)))
	return total_sim

def compute_proto_exemp_probs(rep_set, expt, cg=[C_VAL, G_VAL]):#probability of correct classification according to either models
	n = (len(rep_set)/2) - 1
	samples = expt['block_learning']['all']
	samples = [sample[0] for sample in samples]
	exemplars = {'a':[rep_set[sample] for sample in samples if 'a' in sample],'b':[rep_set[sample] for sample in samples if 'b' in sample]}
	prototypes = {'a':[[term/len(exemplars['a'][0]) for term in [sum([dim[i] for dim in exemplars['a']]) for i in range(len(exemplars['a'][0]))]]],'b':[[term/len(exemplars['b'][0]) for term in [sum([dim[i] for dim in exemplars['b']]) for i in range(len(exemplars['b'][0]))]]]}
	#prototypes = {'a':[rep_set['a0']],'b':[rep_set['b0']]}
	cg = [abs(cg[0]),abs(cg[1])]
	probs = dict()
	for img in rep_set.keys():
		if 'a' in img:
			proto_prob = similarity_images(rep_set[img], prototypes['a'], cg[0])**cg[1]
			proto_prob = proto_prob/(proto_prob+similarity_images(rep_set[img], prototypes['b'], cg[0])**cg[1])
			exemp_prob = similarity_images(rep_set[img], exemplars['a'], cg[0])**cg[1]
			exemp_prob = exemp_prob/(exemp_prob+similarity_images(rep_set[img], exemplars['b'], cg[0])**cg[1])
		else:
			proto_prob = similarity_images(rep_set[img], prototypes['b'], cg[0])**cg[1]
			proto_prob = proto_prob/(proto_prob+similarity_images(rep_set[img], prototypes['a'], cg[0])**cg[1])
			exemp_prob = similarity_images(rep_set[img], exemplars['b'], cg[0])**cg[1]
			exemp_prob = exemp_prob/(exemp_prob+similarity_images(rep_set[img], exemplars['a'], cg[0])**cg[1])
		probs[img] = {'p':proto_prob, 'e':exemp_prob}
	return probs

def generate_histogram(samples_list, bins=20, labels=['lp_rp','lp_re','le_rp','le_re','actual']):
	minval = float('inf')
	maxval = float('-inf')
	n = len(samples_list)
	for samples in samples_list:
		minval = min(minval, min(samples))
		maxval = max(maxval, max(samples))
	binsize = (maxval-minval)/bins
	hists = [[1.0 for i in range(bins)] for j in range(n)]
	for i in range(n):
		for sample in samples_list[i]:
			hists[i][min(int(floor((sample-minval)/binsize)),bins-1)] += 1
	hists = [[i/(len(samples_list[j])+bins) for i in hists[j]] for j in range(n)]
	hist_x = [minval+i*binsize for i in range(bins)]
	for i in range(n):
		try:
			plt.plot(hist_x,hists[i],symbols[i],label=labels[i])
		except:
			try:
				plt.plot(hist_x,hists[i],label=labels[i])
			except:
				plt.plot(hist_x,hists[i])
	plt.xlabel(r'Probability of correct classification: $\theta$')
	plt.ylabel(r'$P(\theta)$')
	plt.title(r'Probability Distribution of the paramater $\theta$')
	plt.legend()
	plt.savefig('paraprobs.png')
	plt.close()
	return (hists, hist_x)

def bayes_evaluation(rep_set, expt): #assuming priors for both hypotheses is same (i.e., 0.5 each)
	parameters = {'l':[],'r':[]}
	hypotheses = ['p','e']
	maxc = {'l':{'p':C_VAL,'e':C_VAL},'r':{'p':C_VAL,'e':C_VAL}}
	maxg = {'l':{'p':G_VAL,'e':G_VAL},'r':{'p':G_VAL,'e':G_VAL}}
	for format in parameters.keys(): #begin comment
		samples = [sample for sample in expt['block_testing'][format] if sample[3] != -1] #ignore unanswered questions
		num = len(samples)
		data = [sample[3] for sample in samples]
		for hypothesis in hypotheses:
			cval = 0.0000000001
			maxval = float('-inf')
			for k in range(9):
				cval *= 10
				gval = 0.00001
				for l in range(5):
					gval *= 10
					computed_probs = compute_proto_exemp_probs(rep_set, expt, [cval, gval])
					probs = [computed_probs[sample[0]] for sample in samples]
					marginal = sum([log(probs[i][hypothesis]) if data[i]==1 else log(1-probs[i][hypothesis]) for i in range(num)])
					if marginal>maxval:
						maxval = marginal
						maxc[format][hypothesis] = cval
						maxg[format][hypothesis] = gval #end comment'''
	for format in parameters.keys():
		samples = expt['block_testing'][format]
		samples = [sample for sample in samples if sample[3] != -1] #ignore unanswered questions
		num = len(samples)
		data = [sample[3] for sample in samples]
		act_param = float(sum(data))/len(data) #actual probability of correct answers
		maxpro = [compute_proto_exemp_probs(rep_set, expt, [maxc[format][hypothesis], maxg[format][hypothesis]]) for hypothesis in hypotheses]
		probs = dict([(hypotheses[i],[maxpro[i][sample[0]] for sample in samples]) for i in range(2)])
		est_param = [[prob[hypothesis] for prob in probs[hypothesis]] for hypothesis in hypotheses]
		est_param = dict([(hypotheses[i],sum(est_param[i])/len(est_param[i])) for i in range(2)])
		marginals = [exp(sum([log(probs[hypothesis][i][hypothesis]) if data[i]==1 else log(1-probs[hypothesis][i][hypothesis]) for i in range(num)])) for hypothesis in hypotheses]
		sum_margs = sum(marginals)
		hypo_prob = dict([(hypotheses[i],marginals[i]/sum_margs) for i in range(2)])
		parameters[format] = {'act_param':act_param,'est_param':est_param,'hypo_prob':hypo_prob}
	return parameters

def bayes_evaluation(rep_set, expt): #assuming priors for both hypotheses is same (i.e., 0.5 each)
	hypotheses = [(('l','p'),('r','p')),(('l','p'),('r','e')),(('l','e'),('r','p')),(('l','e'),('r','e'))]
	maxc = dict([(hypotheses[i],C_VAL) for i in range(4)])
	maxg = dict([(hypotheses[i],G_VAL) for i in range(4)])
	samples = [sample for sample in expt['block_testing']['all'] if sample[3] != -1] #ignore unanswered questions
	num = len(samples)
	data = [sample[3] for sample in samples]
	for hypothesis in hypotheses:
		cval = 0.0000000001
		maxval = float('-inf')
		for k in range(9):
			cval *= 10
			gval = 0.00001
			for l in range(6):
				gval *= 10
				computed_probs = compute_proto_exemp_probs(rep_set, expt, [cval, gval])
				probs = [computed_probs[sample[0]][hypothesis[sample[1]=='r'][1]] for sample in samples]
				marginal = sum([log(probs[i]) if data[i]==1 else log(1-probs[i]) for i in range(num)])
				if marginal>maxval:
					maxval = marginal
					maxc[hypothesis] = cval
					maxg[hypothesis] = gval
	pprint(maxc)
	pprint(maxg)
	print'***********'
	act_param = float(sum(data))/len(data) #actual probability of correct answers
	maxpro = [compute_proto_exemp_probs(rep_set, expt, [maxc[hypothesis], maxg[hypothesis]]) for hypothesis in hypotheses]
	probs = [[maxpro[i][sample[0]][hypotheses[i][sample[1]=='r'][1]] for sample in samples] for i in range(4)]
	est_param = dict([(hypotheses[i],sum(probs[i])/len(probs[i])) for i in range(4)])
	marginals = [exp(sum([log(probs[j][i]) if data[i]==1 else log(1-probs[j][i]) for i in range(num)])) for j in range(4)]
	sum_margs = sum(marginals)
	hypo_prob = dict([(hypotheses[i],marginals[i]/sum_margs) for i in range(4)])
	parameters = {'act_param':act_param,'est_param':est_param,'hypo_prob':hypo_prob}
	return parameters

def calculate_marginal(cg, rep_set, expt, hypothesis, samples, data):
	computed_probs = compute_proto_exemp_probs(rep_set, expt, cg)
	probs = [computed_probs[sample[0]][hypothesis[sample[1]=='r'][1]] for sample in samples]
	num = len(samples)
	try:
		marginal = -sum([log(probs[i]) if data[i]==1 else log(1-probs[i]) for i in range(num)])
	except:
		marginal = float('inf')
	return marginal

def bayes_evaluation(rep_set, expt): #assuming priors for both hypotheses is same (i.e., 0.5 each)
	hypotheses = [(('l','p'),('r','p')),(('l','p'),('r','e')),(('l','e'),('r','p')),(('l','e'),('r','e'))]
	maxcg = dict([(hypotheses[i],[C_VAL,G_VAL]) for i in range(4)])
	samples = [sample for sample in expt['block_testing']['all'] if sample[3] != -1] #ignore unanswered questions
	num = len(samples)
	data = [sample[3] for sample in samples]
	for hypothesis in hypotheses:
		cg = np.array([C_VAL, G_VAL])
		result = minimize(calculate_marginal, cg, args=(rep_set,expt,hypothesis,samples,data), method='Nelder-Mead', options={'xtol': 1e-9, 'disp': True})
		cg = result.x.tolist()
		maxcg[hypothesis] = cg
	pprint(maxcg)
	print'***********'
	act_param = float(sum(data))/len(data) #actual probability of correct answers
	maxpro = [compute_proto_exemp_probs(rep_set, expt, maxcg[hypothesis]) for hypothesis in hypotheses]
	probs = [[maxpro[i][sample[0]][hypotheses[i][sample[1]=='r'][1]] for sample in samples] for i in range(4)]
	est_param = dict([(hypotheses[i],sum(probs[i])/len(probs[i])) for i in range(4)])
	marginals = [exp(sum([log(probs[j][i]) if data[i]==1 else log(1-probs[j][i]) for i in range(num)])) for j in range(4)]
	sum_margs = sum(marginals)
	hypo_prob = dict([(hypotheses[i],marginals[i]/sum_margs) for i in range(4)])
	parameters = {'act_param':act_param,'est_param':est_param,'hypo_prob':hypo_prob}
	return parameters

def to_check_or_not_to_check(expt, prob_thresh):
	for format in ['l','r']:
		samples = expt['block_testing'][format]
		samples = [sample for sample in samples if sample[3] != -1] #ignore unanswered questions
		num = len(samples)
		data = [sample[3] for sample in samples]
		act_param = float(sum(data))/len(data) #actual probability of correct answers
		if act_param<prob_thresh:
			return False
	return True

def to_check_or_not_to_check(expt, prob_thresh):
	samples = expt['block_testing']['all']
	samples = [sample for sample in samples if sample[3] != -1] #ignore unanswered questions
	num = len(samples)
	data = [sample[3] for sample in samples]
	act_param = float(sum(data))/len(data) #actual probability of correct answers
	return act_param>prob_thresh

def low_pass_filter(signal, fraction=0.2):
	signal = np.asarray(signal)
	fft = np.fft.fft(signal)
	fft = [fft[i] if i<fraction*len(fft) else 0 for i in range(len(fft))]
	out = np.ndarray.tolist(np.abs(np.fft.ifft(fft)))
	return out

def plot_reaction_times(expt, subject_code, hypothesis):
	rt = [[sample[4] for sample in expt['block_contslearn'][format]] for format in ['l','r']]
	labels=['left','right']
	symbols=['r','b']
	for i in range(2):
		plt.plot(low_pass_filter(rt[i]),symbols[i],label=labels[i])
	plt.legend()
	plt.ylabel('Reaction Time (milliseconds)')
	plt.xlabel('Sample Number')
	hyponame = "".join(hypothesis[0])+'_'+"".join(hypothesis[1])
	plt.title('Variation of reaction time for Subject '+str(subject_code)+', Model: '+hyponame)
	plt.savefig(hyponame+'_'+str(subject_code)+'.png')
	plt.close()

def find_optimal_hypotheses(subject_params, subjects):
	num_subjects = {'l_p':0,'l_e':0,'r_p':0,'r_e':0}
	for subject in subject_params.keys():
		for format in ['l','r']:
			prob_p = subject_params[subject][format]['hypo_prob']['p']
			prob_e = subject_params[subject][format]['hypo_prob']['e']
			'''if prob_p > prob_e:
				num_subjects[format+'_p'] += 1
			else:
				num_subjects[format+'_e'] += 1'''
			if prob_p - prob_e > HYPO_DIFF_THRESH:
				num_subjects[format+'_p'] += 1
				print(subject, format, prob_p, prob_e)
				plot_reaction_times(parse_exp(subjects[subject]['exp2file']), subject)
			elif prob_e - prob_p > HYPO_DIFF_THRESH:
				num_subjects[format+'_e'] += 1				
				print(subject, format, prob_p, prob_e)
				plot_reaction_times(parse_exp(subjects[subject]['exp2file']), subject)
	samples_list = [[subject_params[subject][format]['hypo_prob'][model] for subject in subject_params.keys()] for (format,model) in [('l','p'),('l','e'),('r','p'),('r','e')]]
	labels=['left-prototype','left-exemplar','right-prototype','right-exemplar']
	symbols=['ro','go','g^','r^']
	for i in range(len(samples_list)):
		plt.plot(samples_list[i],symbols[i],label=labels[i])
	plt.legend()
	plt.ylabel('Probability of Hypothesis')
	plt.xlabel('Subject')
	plt.title('Probabilities of Hypotheses for the filtered subjects')
	plt.show()
	pprint(num_subjects)
	return num_subjects

def find_optimal_hypotheses(subject_params, subjects):
	num_subjects = {(('l','p'),('r','p')):0,(('l','p'),('r','e')):0,(('l','e'),('r','p')):0,(('l','e'),('r','e')):0}
	for subject in subject_params.keys():
		'''for hypothesis in num_subjects.keys():
			num_subjects[hypothesis] += subject_params[subject]['hypo_prob'][hypothesis]'''
		vals = [subject_params[subject]['hypo_prob'][hypothesis] for hypothesis in num_subjects.keys()]
		maxval = max(vals)
		if maxval>HYPO_DIFF_THRESH:
			for i in range(len(vals)):
				if vals[i]==maxval:
					hypotheses = num_subjects.keys()
					num_subjects[hypotheses[i]] += 1
					plot_reaction_times(parse_exp(subjects[subject]['exp2file']), subject, hypotheses[i])
	pprint(subject_params)
	samples_list = [[subject_params[subject]['hypo_prob'][hypothesis] for subject in subject_params.keys()] for hypothesis in num_subjects.keys()]
	accuracies = [subject_params[subject]['act_param'] for subject in subject_params.keys()]
	label_dict = {(('l','p'),('r','p')):'lp-rp',(('l','p'),('r','e')):0,(('l','e'),('r','p')):0,(('l','e'),('r','e')):0}
	labels=[hypothesis[0][0]+hypothesis[0][1]+'_'+hypothesis[1][0]+hypothesis[1][1] for hypothesis in num_subjects.keys()]
	symbols=['ro','go','bo','co']
	for i in range(len(samples_list)):
		plt.plot(accuracies,samples_list[i],symbols[i],label=labels[i])
	plt.legend()
	plt.ylabel('Probability of Hypothesis')
	plt.xlabel('Subject Accuracy')
	plt.title('Probabilities of Hypotheses for the filtered subjects')
	plt.savefig('hypoprobs.png')
	plt.close()
	pprint(num_subjects)
	return num_subjects

def average_analysis(subject_params):
	samples_list = [[subject_params[subject][format]['est_param'][model] for subject in subject_params.keys()] for (format,model) in [('l','p'),('l','e'),('r','p'),('r','e')]] + [[subject_params[subject][format]['act_param'] for subject in subject_params.keys()] for format in ['l','r']]
	(hists, hist_x) = generate_histogram(samples_list)
	kls = {'l':{'p':0,'e':0},'r':{'p':0,'e':0}}
	map = {'lp':0,'le':1,'rp':2,'re':3,'l':4,'r':5}
	for format in ['l','r']:
		for model in ['p','e']:
			kls[format][model] = kl_divergence(hists[map[format+model]],hists[map[format]])
	pprint(kls)

def average_analysis(subject_params):
	hypotheses = [(('l','p'),('r','p')),(('l','p'),('r','e')),(('l','e'),('r','p')),(('l','e'),('r','e'))]
	samples_list = [[subject_params[subject]['est_param'][hypothesis] for subject in subject_params.keys()] for hypothesis in hypotheses] + [[subject_params[subject]['act_param'] for subject in subject_params.keys()]]
	(hists, hist_x) = generate_histogram(samples_list)
	map = dict([(hypotheses[i],i) for i in range(len(hypotheses))])
	kls = dict([(hypotheses[i],0.0) for i in range(len(hypotheses))])
	for hypothesis in hypotheses:
		kls[hypothesis] = kl_divergence(hists[map[hypothesis]],hists[-1])
	pprint(kls)

def test_params(dat_folders, img_file, prob_thresh=P_THRESH): #better than chance selection
	subjects = parse_csv(dat_folders)
	rep_set = parse_img(img_file)
	subject_params = dict()
	print 'Initially', len(subjects), 'confident subjects are shortlisted'
	print 'Selecting subjects with probability of correct guess in exp1 no less than', prob_thresh
	for subject in subjects.keys():
		expt = parse_exp(subjects[subject]['exp1file'])
		if to_check_or_not_to_check(expt, prob_thresh):
			#try:
				subject_params[subject] = bayes_evaluation(rep_set, expt)
				print 'Selecting',subject
			#except:
				print 'Ignoring',subject,'; penalise'
		else:
			print 'Ignoring',subject,'for poor accuracy'
	print 'Selected', len(subject_params), 'rightly confident subjects for parameter estimation and computation'
	find_optimal_hypotheses(subject_params, subjects)
	average_analysis(subject_params)
	return subject_params

def kl_divergence(dist1, dist2): #computes Kullback-Leibler divergence between two discrete probability distributions (=0 for identical, calculated of 1 wrt 2)
	return sum([dist1[i]*log(dist1[i]/dist2[i],2) for i in range(len(dist1))])

expmt_data_folders = ['../data/day1','../data/remaining']
imgs1_file = '../data/rep_set_1.dat'
imgs2_file = '../data/rep_set_2.dat' #not required, really

a = test_params(expmt_data_folders, imgs1_file)
#pprint(a)
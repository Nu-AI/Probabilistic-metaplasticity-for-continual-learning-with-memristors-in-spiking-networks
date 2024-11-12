# split-Fashion MNIST classification with memristor weights

import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
import multiprocessing
from tqdm import notebook
import json
import copy
from tqdm import trange, tqdm
from multiprocessing import Pool, RLock
import tensorflow as tf
import keras
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

def res_param_config(mean_res, std_res, n_cross, w_hid_max, w_out_max):
	# creating the array with n_cross res states
	states = []
	for i in range(len(mean_res)-1):
		temp = np.ones(n_cross)*mean_res[i]
		for j in range(n_cross):
			temp1 = copy.deepcopy(temp)
			states.append(temp1)
			temp[n_cross-j-1] = mean_res[i+1]
	states.append(np.ones(n_cross)*mean_res[len(mean_res)-1])

	# calculate the parallel equivalent resistance
	p = []
	for state in states:
		temp = 0
		for j in range(len(state)):
			temp = temp + 1/state[j]
		p.append(1/temp)

	# compute the feedback and bias resistance for given weight range
	b_h = np.array([[w_hid_max], [-w_hid_max]])
	b_o = np.array([[w_out_max], [-w_out_max]])
	A = np.array([[1/p[len(p)-1], -1], [1/p[0], -1]])
	x_h = np.matmul(np.linalg.inv(A), b_h)
	x_o = np.matmul(np.linalg.inv(A), b_o)

	R_fh = x_h[0][0]
	R_bh = R_fh/x_h[1][0]
	R_fo = x_o[0][0]
	R_bo = R_fo/x_o[1][0]
	
	return R_fh, R_bh, R_fo, R_bo
	
def res_to_weight(r, R_f, R_b):   
	max_axis = len(r.shape)-1
	r_tot = 1/(np.sum(1/r, axis = max_axis))    
	weight = R_f/r_tot - R_f/R_b
	return weight


def weight_initialize_var(n1, n2, R_f, R_b, n_cross, w_max):
	states = []
	statesP = []
	for i in range(len(mean_res)-1):
		temp = np.ones(n_cross)*mean_res[i]
		tempP = np.ones(n_cross)*i
		for j in range(n_cross):
			temp1 = copy.deepcopy(temp)
			temp1P = copy.deepcopy(tempP)
			states.append(temp1)
			statesP.append(temp1P)
			temp[n_cross-j-1] = mean_res[i+1]
			tempP[n_cross-j-1] = i+1
	states.append(np.ones(n_cross)*mean_res[len(mean_res)-1])
	statesP.append(np.ones(n_cross)*(len(mean_res)-1))
	states = np.array(states)
	statesP = np.array(statesP)             
	w_list = res_to_weight(states, R_f, R_b)
	n_tot = n1*n2
	n_ind1 = np.where((w_list<=-0.5)&(w_list>=-1))
	n_ind2 = np.where((w_list<0)&(w_list>=-0.5))
	n_ind3 = np.where((w_list>=0)&(w_list<0.5))
	n_ind4 = np.where((w_list<=1)&(w_list>=0.5))
	s1 = numpy.random.choice( n_ind1[0] , size = int(n_tot/4), replace = True, p = None)
	s2 = numpy.random.choice( n_ind2[0] , size = int(n_tot/4), replace = True, p = None)
	s3 = numpy.random.choice( n_ind3[0] , size = int(n_tot/4), replace = True, p = None)
	s4 = numpy.random.choice( n_ind4[0] , size = int(n_tot/4), replace = True, p = None)
	ind = numpy.concatenate ((s1, s2, s3, s4), axis = 0, out = None)
	ind_rand = numpy.random.choice( ind , size = int(n_tot), replace = False, p = None)
	r = np.zeros((n_tot, n_cross))
	rP = np.zeros((n_tot, n_cross))
	rP = statesP[ind_rand]
	for i in range(n_res_level):
		loc = np.where(rP==i)
		if len(loc[0])!=0:
			r[loc] = np.random.normal(mean_res[i], std_res[i], len(loc[0]))
	r = np.reshape(r, [n1, n2, n_cross])
	rP = np.reshape(rP, [n1, n2, n_cross])
	w = res_to_weight(r, R_f, R_b)
	return w, r
	
def data_load():
	((X_train, TrainL_), (X_test, TestL_)) = fashion_mnist.load_data()
	X_train=X_train/255
	X_test=X_test/255
	TrainIm_= np.reshape(X_train, [X_train.shape[0],X_train.shape[1]*X_train.shape[2]])
	TestIm_ = np.reshape(X_test, [X_test.shape[0],X_test.shape[1]*X_test.shape[2]])
	return TrainIm_, TrainL_, TestIm_, TestL_
	
def infer_level(r_up):
	temp_res = np.matlib.repmat(np.reshape(r_up, (len(r_up),1)),1, n_res_level)
	diff = abs(temp_res-mean_res)
	inferred_level = np.argmin(diff, axis =1)
	return inferred_level

def res_program(r, up_dir):
	r_P = infer_level(r)
	r_P = r_P + up_dir
	r_P[np.where(r_P > len(mean_res)-1)] = len(mean_res)-1
	r_P[np.where(r_P < 0)] = 0
	r = np.zeros_like(r_P)
	for i in range(n_res_level):
		loc = np.where(r_P==i)
		r[loc] = np.random.normal(mean_res[i], std_res[i], len(loc[0]))
	return r 
	
def make_spike_trains(freqs, n_steps):
	''' Create an array of Poisson spike trains
		Parameters:
			freqs: Array of mean spiking frequencies.
			n_steps: Number of time steps
	'''
	r = np.random.rand(len(freqs), n_steps)
	spike_trains = np.where(r <= np.reshape(freqs, (len(freqs),1)), 1, 0)
	return spike_trains

def MNIST_to_Spikes(maxF, im, t_sim, dt):
	''' Generate spike train array from MNIST image.
		Parameters:
			maxF: max frequency, corresponding to 1.0 pixel value
			FR: MNIST image (784,)
			t_sim: duration of sample presentation (seconds)
			dt: simulation time step (seconds)
	'''
	n_steps = int(t_sim / dt) #  sample presentation duration in sim steps
	freqs = im * maxF * dt # scale [0,1] pixel values to [0,maxF] and flatten
	SpikeMat = make_spike_trains(freqs, n_steps)
	return SpikeMat
	
class NumpyEncoder(json.JSONEncoder):
	""" Special json encoder for numpy types """
	def default(self, obj):
		if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
							np.int16, np.int32, np.int64, np.uint8,
							np.uint16, np.uint32, np.uint64)):
			return int(obj)
		elif isinstance(obj, (np.float_, np.float16, np.float32,
							  np.float64)):
			return float(obj)
		elif isinstance(obj, (np.ndarray,)):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)
		
def check_accuracy(images, labels, w_in, w_out):
	"""Present a set of labeled images to the network and count correct inferences
	:param images: images
	:param labels: labels
	:return: fraction of labels correctly inferred
	"""
	numCorrect = 0

	for u in range(len(images)):
		cnt = np.zeros(n_out)
		spikeMat = MNIST_to_Spikes(MaxF, images[u], tSim, dt_conv)

		# Initialize hidden layer variables
		I1 = np.zeros(n_h1)
		V1 = np.zeros(n_h1)

		# Initialize output layer variables
		I2 = np.zeros(n_out)
		V2 = np.zeros(n_out)

		# Initialize firing time variables
		ts1 = np.full(n_h1, -t_refr)
		ts2 = np.full(n_out, -t_refr)


		for t in range(nBins):
			# Update hidden neuron synaptic currents
			I1 += (dt/t_syn) * (w_in.dot(spikeMat[:, t]) - I1)

			# Update hidden neuron membrane potentials
			V1 += (dt/t_m) * ((V_rest - V1) + I1 * R)
			V1[V1 < -Vth/10] = -Vth/10 # Limit negative potential to -Vth/10

			# Clear membrane potential of hidden neurons that spiked more
			# recently than t_refr
			V1[t*dt - ts1 <= t_refr] = 0

			## Process hidden neuron spikes
			fired = np.nonzero(V1 >= Vth) # Hidden neurons that spiked
			V1[fired] = 0 # Reset their membrane potential to zero
			ts1[fired] = t # Update their most recent spike times

			# Make array of hidden-neuron spikes
			ST1 = np.zeros(n_h1)
			ST1[fired] = 1

			# Update output neuron synaptic currents
			I2 += (dt/t_syn1)*(w_out.dot(ST1) - I2)

			# Update output neuron membrane potentials
			V2 += (dt/t_mH)*((V_rest - V2) + I2*(RH))
			V2[V2 < -VthO/10] = -VthO/10 # Limit negative potential to -Vth0/10

			# Clear V of output neurons that spiked more recently than t_refr
			refr2 = (t*dt - ts2 <= t_refr)
			V2[refr2] = 0

			## Process output spikes
			fired2 = np.nonzero(V2 >= VthO) # output neurons that spikes
			V2[fired2] = 0 # Reset their membrane potential to zero
			ts2[fired2] = t # Update their most recent spike times

			# Make array of output neuron spikes
			ST2 = np.zeros(n_out)
			ST2[fired2] = 1

			cnt += ST2

		if np.count_nonzero(cnt) != 0:  # Avoid counting no spikes as predicting label 0
			prediction = np.argmax(cnt)
			target = labels[u]
			if prediction == target:
				numCorrect += 1

	return numCorrect/len(images)
	
def mem_class_train (params):
	ind_ = params['ind']
	seed = params["seed"]
	U_in = params["U_in"]
	U_out = params["U_out"]
	np.random.seed(seed)
	Acc = np.zeros((n_tasks,n_tasks,n_runs))
	for run in range(n_runs):

		# Randomly select train and test samples
		trainInd = np.random.choice(len(TrainIm_), n_train, replace=False)
		TrainIm = TrainIm_[trainInd]
		TrainLabels = TrainL_[trainInd]

		testInd = np.random.choice(len(TestIm_), n_test, replace=False)
		TestIm = TestIm_[testInd]
		TestLabels = TestL_[testInd]



		# Generate random feedback weights
		w_err_factor = 0.15
		w_err_h1p = ((np.random.rand(n_h1,n_out))*2-1)*w_err_factor # these are random numbers from -1 to 1
		w_err_h1n = w_err_h1p
		
		ttt = []
		for dd in range(n_tasks):
			temp_trainInd = np.concatenate((np.where(TrainLabels == taskID[dd,0])[0],np.where(TrainLabels == taskID[dd,1])[0]),axis=0)
			ttt.append(len(temp_trainInd))
		
		with tqdm(total=n_tasks*maxE*int(np.mean(ttt))*nBins,desc="Run {} of params index {}".format(run,ind_),position=ind_) as pbar:
			
			cross_ind_in = 0 
			cross_ind_out = 0
			for d in range(n_tasks): #n_tasks

				# Generate forward pass weights
				w_in, r_in = weight_initialize_var(n_h1, n_in, R_fh, R_bh, n_cross, w_in_max)
				w_out, r_out = weight_initialize_var(n_out, n_h1, R_fo, R_bo, n_cross, w_out_max)

				trainInd = np.concatenate((np.where(TrainLabels == taskID[d,0])[0],np.where(TrainLabels == taskID[d,1])[0]),axis=0)
				n_train2 = len(trainInd)
				trainInd2 = np.random.choice(len(trainInd), n_train2, replace=False)
				trainInd = trainInd[trainInd2]
				taskLabels = TrainLabels[trainInd]
				trainSet = TrainIm[trainInd]
				taskID2 = np.where(taskLabels == taskID[d,1])
				taskLabelsF = np.zeros(len(trainInd));
				taskLabelsF[taskID2] = 1

				n_train2 = len(trainInd)
				for e in range(maxE):
					for u in range(n_train2):
					
						im = trainSet[u]
						fr = im*MaxF
						spikeMat = MNIST_to_Spikes(MaxF, trainSet[u], tSim, dt_conv)
						fr_label = np.zeros(n_out)
						fr_label[int(taskLabelsF[u])] = maxFL # target output spiking frequencies
						s_label = make_spike_trains(fr_label*dt_conv, nBins) # target spikes


						# Initialize hidden layer variables
						I1 = np.zeros(n_h1)
						V1 = np.zeros(n_h1)
						U1 = np.zeros(n_h1)
						
						
						# Initialize output layer variables
						I2 = np.zeros(n_out)
						V2 = np.zeros(n_out)
						U2 = np.zeros(n_out)
						
						
						# Initialize error neuron variables
						Verr1 = np.zeros(n_out)
						Verr2 = np.zeros(n_out)

						# Initialize firing time variables
						ts1 = np.full(n_h1, -t_refr)
						ts2 = np.full(n_out, -t_refr)

						for t in range(nBins):
							# Forward pass

							# Find input neurons that spike
							fired_in = np.nonzero(spikeMat[:, t])

							# Update synaptic current into hidden layer
							I1 += (dt/t_syn) * (w_in.dot(spikeMat[:, t]) - I1)

							# Update hidden layer membrane potentials
							V1 += (dt/t_m) * ((V_rest - V1) + I1 * R)
							V1[V1 < -Vth/10] = -Vth/10 # Limit negative potential

							# If neuron in refractory period, prevent changes to membrane potential
							refr1 = (t*dt - ts1 <= t_refr)
							V1[refr1] = 0

							fired = np.nonzero(V1 >= Vth) # Hidden neurons that spiked
							V1[fired] = 0 # Reset their membrane potential to zero
							ts1[fired] = t # Update their most recent spike times

							ST1 = np.zeros(n_h1) # Hidden layer spiking activity
							ST1[fired] = 1 # Set neurons that spiked to 1
							
							# Repeat the process for the output layer
							I2 += (dt/t_syn1)*(w_out.dot(ST1) - I2)

							V2 += (dt/t_mH)*((V_rest - V2) + I2*(RH))
							V2[V2 < -VthO/10] = -VthO/10

							refr2 = (t*dt - ts2 <= t_refr)
							V2[refr2] = 0
							fired2 = np.nonzero(V2 >= VthO)

							V2[fired2] = 0
							ts2[fired2] = t

							# Make array of output neuron spikes
							ST2 = np.zeros(n_out)
							ST2[fired2] = 1
							

							# Compare with target spikes for this time step
							Ierr = (ST2 - s_label[:, t])

							# Update false-positive error neuron membrane potentials
							Verr1 += (dt/t_mE)*(Ierr*RE)
							Verr1[Verr1 < -VthE/10] = -VthE/10 # Limit negative potential to -VthE/10

							## Process spikes in false-positive error neurons
							fired_err1 = np.nonzero(Verr1 >= VthE)
							Verr1[fired_err1] -= VthE

							# Don't penalize "false positive" spikes on the target
							Verr1[int(taskLabelsF[u])] *= FPF

							# Make array of false-positive error neuron spikes
							Serr1 = np.zeros(n_out)
							Serr1[fired_err1] = 1

							# Update false-negative error neuron membrane potentials
							Verr2 -= (dt/t_mE)*(Ierr*RE)
							Verr2[Verr2 < -VthE/10] = -VthE/10

							## Process spikes in false-negative error neurons
							fired_err2 = np.nonzero(Verr2 >= VthE)
							Verr2[fired_err2] -= VthE


							# Make array of false-negative error neuron spikes
							Serr2 = np.zeros(n_out)
							Serr2[fired_err2] = 1


							# Update hidden neuron error compartments (using random weights)
							U1 += (dt/t_mU)*( (w_err_h1p.dot(Serr1) - w_err_h1n.dot(Serr2))*RU)

							# Update output neuron error compartments
							U2 += (dt/t_mU)*( (Serr1 - Serr2)*RU)

							up_hid = np.where(np.abs(U1)>U_in)[0]
							up_out = np.where(np.abs(U2)>U_out)[0]
							
							
							if len(up_hid)>0: # if any neuron error has passed threshold
								post_ind = np.nonzero((I1[up_hid]>Imin) & (I1[up_hid]<Imax))[0]
								if len(post_ind)>0:
									if len(fired_in[0]) != 0:
										s = np.sign(U1[up_hid[post_ind]])  # getting the sign of the errors, it is 1 if U2 positive, -1 if if U2 negative, 0 if U2 is zero
										U1[up_hid[post_ind]] = 0
										pre_ind = fired_in  # collects the location of the pre-synaptic neurons										
										UF = np.matlib.repmat(np.reshape(s, (len(s),1)),1, len(pre_ind[0])) # extends the error
										c_in = np.zeros([n_h1,n_in])
										c_in[np.ix_(up_hid[post_ind], pre_ind[0])] -= UF 
										up_in_mem = np.where(c_in!=0)
										if len(up_in_mem[0])>0:											
											cross_ind_in = cross_ind_in+1 
											current_ind = int(cross_ind_in%n_cross)
											c_up = c_in[up_in_mem]
											r_in_up = r_in[up_in_mem][:,current_ind]
											r_in[up_in_mem[0], up_in_mem[1],current_ind] = res_program(r_in_up, c_up)
											w_in[up_in_mem]  = res_to_weight(r_in[up_in_mem], R_fh, R_bh)
											c_in = np.zeros([n_h1,n_in])
										


							if len(up_out)>0: # if any neuron error has passed threshold
								post_ind = np.nonzero((I2[up_out]>Imin) & (I2[up_out]<Imax))[0]
								if len(post_ind)>0:	
									if len(fired[0]) != 0:
										s = np.sign(U2[up_out[post_ind]]) 
										U2[up_out[post_ind]] = 0
										pre_ind = fired  # collects the location of the pre-synaptic neurons
										UF = np.matlib.repmat(np.reshape(s, (len(s),1)),1, len(pre_ind[0])) # extends the error
										c_out = np.zeros([n_out, n_h1]) 
										c_out[np.ix_(up_out[post_ind], pre_ind[0])] -= UF 
										up_out_mem = np.where(c_out!=0)				
										if len(up_out_mem[0])>0:
											cross_ind_out = cross_ind_out+1 
											current_ind = int(cross_ind_out%n_cross)
											c_up = c_out[up_out_mem]
											r_out_up = r_out[up_out_mem][:,current_ind]
											r_out[up_out_mem[0], up_out_mem[1],current_ind] = res_program(r_out_up, c_up)
											w_out[up_out_mem] = res_to_weight(r_out[up_out_mem], R_fo, R_bo)
											c_out = np.zeros([n_out, n_h1])  

							pbar.update(1)

						

							


			

				for d2 in range(d+1):

					testInd = np.concatenate((np.where(TestLabels == taskID[d2,0])[0],np.where(TestLabels == taskID[d2,1])[0]),axis=0)
					taskLabels = TestLabels[testInd]
					testSet = TestIm[testInd]
					taskID2 = np.where(taskLabels == taskID[d2,1])
					taskLabelsT = np.zeros(len(testInd))
					taskLabelsT[taskID2] = 1

					Acc[d2, d, run] = check_accuracy(testSet, taskLabelsT, w_in, w_out )
					

				



	avg_task_acc = np.mean(Acc,axis=2)
	avg_task_std = np.std(Acc,axis=2)
	class_cont_Acc= np.zeros(n_tasks)
	class_cont_std =np.zeros(n_tasks)
	for i in range(n_tasks):
		class_cont_Acc[i] = avg_task_acc[i,n_tasks-1]
		class_cont_std[i] = avg_task_std[i,n_tasks-1]
		
	class_Acc= np.zeros(n_tasks)
	class_std =np.zeros(n_tasks)
	for i in range(n_tasks):
		class_Acc[i] = avg_task_acc[i,i]
		class_std[i] = avg_task_std[i,i]
	cont_acc = np.mean(Acc,axis=0)[n_tasks-1]
	
	
	mean_cont_acc = np.mean(cont_acc)
	std_cont_acc = np.std(cont_acc)

	results = {'U_in': U_in, 'U_out': U_out, 'class_Acc':class_Acc, 'class_std':class_std, 'class_cont_Acc':class_cont_Acc, 'class_cont_std':class_cont_std, 'cont_mean' : mean_cont_acc, 'cont_std' : std_cont_acc,  'Acc' : Acc, 'seed': seed}

	jsonString = json.dumps(results, indent=4, cls=NumpyEncoder)
	s1 = "U_in_"
	s2 = str(U_in)
	s2.replace(".","_")
	s3 = "U_out_"
	s4 = str(U_out)
	s4.replace(".","_")
	name= "fmnist_prbacc_"+s1+s2+s3+s4
	filename = "%s.json" % name
	jsonFile = open(filename, "w")
	jsonFile.write(jsonString)
	jsonFile.close()


	return results
	

#weight parameters
lr_factor = 7
w_in_max = 3
w_out_max = 1.5

current_path = os.path.abspath(os.path.dirname(__file__))
file_path = os.path.join(current_path, "HfOx_device_data_placeholder.csv")

memristor_data = pd.read_csv(file_path)

mean_res = np.array(memristor_data["Resistance_level_mean"])
std_res = np.array(memristor_data["Resistance_level_std"])

# The device data has been collected from Liehr, Maximilian, et al. "Impact of switching variability of 65nm CMOS integrated hafnium 
#dioxide-based ReRAM devices on distinct level operations." 2020 IEEE International Integrated Reliability Workshop (IIRW). IEEE, 2020.

n_res_level = len(mean_res)
n_cross = 2
R_fh, R_bh, R_fo, R_bo = res_param_config(mean_res, std_res, n_cross, w_in_max, w_out_max)

# task parameters
n_train = 60000
n_test = 10000
maxE = 1

n_runs = 5
n_tasks = 5
taskID = np.array([[0, 1], [2, 3], [4, 5], [6,7], [8, 9]])

#Learning rule parameters
Imin = -3
Imax = 3
lr0 = 0.1*lr_factor
lr1 = 1e-3*lr_factor
w_scale0 = 1e-0 # Weight scale in hidden layer
w_scale1 = 1e-0 # Weight scale at output layer
FPF = 1 # inhibits punshing target neuron (only use if training a specific output spike pattern)
     


# Simulation parameters
tSim = 0.15 # Duration of simulation (seconds)
MaxF = 250
maxFL = 100
dt = 1 # time resolution
dt_conv = 1e-3 # Data is sampled in ms
nBins = int(tSim/dt_conv) #total no. of time steps

# Network architecture parameters
n_h1 = 200  # no. of hidden neurons
dim = 28 # dim by dim is the dimension of the input images
n_in = dim*dim  # no. of input neurons
n_out = 2   # no. of output neurons 
nTrials = n_in


# Neuron parameters
t_syn = 10
t_syn1 = 25
t_m = 15
t_mH = 25
t_mU = 15
t_mE = 10
t_tr = 25
R = 1
RH = 5
RU = 5
RE = 25
Vs = 15
VsO = 10
VsE = 1
V_rest = 0 # Resting membrane potential
t_refr = 4 # Duration of refractory period

Vth = (1/t_m)*R*Vs # Hidden neuron threshold
VthO = (1/t_mH)*RH*VsO # Output neuron threshold
VthE = (1/t_mE)*RE*VsE # Error neuron threshold


seeds = []
n_parallel = 5

U_inL = [0.45] 
U_outL =  [ 2.25]


TrainIm_, TrainL_, TestIm_, TestL_ = data_load()


ind_ = 0

params = []
for i in U_inL:
	for j in U_outL:
		params.append({'ind':ind_, 'U_in':i, 'U_out':j, "seed" : 2})
		ind_+=1


if __name__ == '__main__':

	tqdm.set_lock(RLock()) 
	p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),),processes = int(multiprocessing.cpu_count()/8))
	p.map(mem_class_train, params)
	p.close()
	p.join()

# installing commands
# pip install -U numpy scipy scikit-learn
# for features, download from github

import scipy.io.wavfile as wav
from features import mfcc
from features import logfbank
from sklearn.cluster import KMeans
import pickle
import os
from os import listdir
from os import walk
from os.path import isfile, join
from shutil import copy2
from sklearn import preprocessing


base_dir = r'.'
mypath = r'./inputs'
outdir = r'./outputs/mfcc_files' 

#Function to get all the wav files given a root directory
def get_files_list(root):
        files_list = []
        for dirpath, dirnames, files in walk(root):
                for name in files:
                        if name.lower().endswith(".wav"):
                                files_list.append(join(dirpath,name))
        return files_list   


def generate_testing_mfccs(myfiles, outdir): 
    for f in myfiles:
        print "Generating MFCCs for: ", f
        (rate, sig) = wav.read(f)
        #print rate, sig.shape
        #get the spectral vectors
        mfcc_feat = mfcc(sig,rate)
        #mfcc_feat = scaler.transform(mfcc_feat)
        #fcomps = os.path.split(f) #file components path, filename
        fcomps = f.split("/")
        fn = fcomps[-2]+"/"+fcomps[-1].split('.')[0] + '_mfcc.txt'
        #outpath = os.path.join(fcomps[0], 'outputs')
        fn = os.path.join(outdir, fn)
        d = os.path.dirname(fn)
        if not os.path.exists(d):
                os.makedirs(d)
        f = open(fn, 'wb')
        final_mfccs_str=""
        for vector in mfcc_feat:
			str_mfcc = ""
			for element in vector:
				str_mfcc +=str(element)+","
			str_mfcc = str_mfcc[:-1]
			final_mfccs_str += str_mfcc+"\n"
        f.write(final_mfccs_str)
        f.close()
        print 'output MFCC file:  ', f, ' written'
    return

    
if __name__ == '__main__':
    
    print("Ensure that you have all the wav files in 'inputs' folder\n")
    myfiles = get_files_list(mypath)
    #generate_training_mfccs(trg_file,outdir)
    generate_testing_mfccs(myfiles, outdir)
    print("Check the 'outputs/mfcc_files' folder for output.")
    
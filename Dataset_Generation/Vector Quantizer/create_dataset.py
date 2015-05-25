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



base_dir = r'.'
trg_file = r'./kmeans_train.wav'
mypath = r'./inputs'
outdir = r'./outputs/vqfiles'


#Function to get all the wav files given a root directory
def get_files_list(root):
        files_list = []
        for dirpath, dirnames, files in walk(root):
                for name in files:
                        if name.lower().endswith(".wav"):
                                files_list.append(join(dirpath,name))
        return files_list   


def build_codebook(trgfile, codesize = 32, fname = None): # given a training file constructs the codebook using kmeans
    (rate, sig) = wav.read(trgfile)
    print rate, sig.shape
    #get the spectral vectors
    print("MFCC generation begins")
    mfcc_feat = mfcc(sig,rate)
    print("MFCC generation ends")
    print mfcc_feat.shape
    print("Fbank creation begins")
    fbank_feat = mfcc_feat #logfbank(sig,rate) #this has the spectral vectors now
    print("Fbank creation ends")
    print fbank_feat.shape
    print "codesize = ", codesize
    km = KMeans(n_clusters = codesize)
    km.fit(fbank_feat)
    if fname != None:
        pickle.dump(km, open(fname, 'wb'))
    return km

def vector_quantize(myfiles, outdir, model): #given a list of files transform them to spectral vectors and compute the KMeans VQ
    for f in myfiles:
        print "Quantizing: ", f
        (rate, sig) = wav.read(f)
        #print rate, sig.shape
        #get the spectral vectors
        mfcc_feat = mfcc(sig,rate)
        #print mfcc_feat.shape
        fbank_feat = mfcc_feat #logfbank(sig,rate) #this has the spectral vectors now
        #print fbank_feat.shape
        val = model.predict(fbank_feat)
        #fcomps = os.path.split(f) #file components path, filename
        fcomps = f.split("/")
        fn = fcomps[-2]+"/"+fcomps[-1].split('.')[0] + '_vq.txt'
        #outpath = os.path.join(fcomps[0], 'outputs')
        fn = os.path.join(outdir, fn)
        d = os.path.dirname(fn)
        if not os.path.exists(d):
                os.makedirs(d)
        #print fn

        #val = trim_background(val)
        #raw_input("enter...")
        
        f = open(fn, 'wb')
        for v in val:
            f.write(str(v) + '\n')
        f.close()
        print 'output vector quantized file:  ', f, ' written'
    return

if __name__ == '__main__':
    fn = 'kmeans.p'
    print("\nEnsure that you have a file named 'kmeans_train.wav' in the current directory if you want to train the k-means classifier.")
    print("Ensure that you have all the directories of the students in the 'inputs' folder\n")
    train_code = int(input("Enter 1 to train the kmeans classifier or 0 to use the pickled file "))
    
    if train_code == 1:
        print("Training begins")
        codebook_size = int(input("Enter the codebook size for vector quantization "))
        km = build_codebook(trg_file, codesize = codebook_size, fname = fn)
        print("Training done")
    km1 = pickle.load(open(fn, 'rb'))
    #print km1.labels_[:100]
    myfiles = get_files_list(mypath)
    #print myfiles
    vector_quantize(myfiles, outdir, km1)
    print("Check the 'outputs/vqfiles' folder for output.")
    
    # now we have everything we need to start using the HMM

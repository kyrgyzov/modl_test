import sys
import numpy as np
import argparse
import os
import psutil
import time
from scipy.sparse import csr_matrix, find
from modl.decomposition.dict_fact import DictFact
#from modl.dict_fact import DictFact


# Parser for passing arguments easily to script
parser = argparse.ArgumentParser(description="Run Online Matrix Factorization as in"
                                             "(https://hal.archives-ouvertes.fr/hal-01431618v2/document)"
                                             "on the human gut samples x hashed k-mers matrices")
parser.add_argument("-d", "--data_dir", required=True,
                    help="directory containing the data (.count.hash.nzi, etc.).")
parser.add_argument("-c", "--n_comp", type=int, default=500,
                    help="number of components (so-called eigen-genomes)")
parser.add_argument("--sps_egen", type=float, default=0.5,
                    help="sparsity parameter for components, 1 : l1-norm, 0 : l2-norm"
                    " Sparse matrix (sps_egen=1) means a bio. sample should be"
                    " expressed as a combination of few different eigen-genomes"
                    " (compared to total number of them, n_comp)")
parser.add_argument("--sps_kmers", type=float, default=0.5,
                    help="sparsity parameter for code matrix, 1 : l1-norm, 0 : l2-norm."
                    " Sparse matrix (sps_kmers=1) means an eigen-genome should"
                    " be expressed as a combination of few different hkmers"
                    " (compared to total number, 2**n_hpp)")

# Functions to check memory and time usage (unused at the moment)

def printt(s):
    """ prints string s with current date and time """
    tms = time.strftime("%d/%m/%Y -- %H:%M:%S")
    print(tms, s)

# Parse arguments
args = parser.parse_args()
data_dir = args.data_dir

res_dir = data_dir + 'res_fact/'
# Create result directory if non existing yet
if not os.path.exists(res_dir):
    os.mkdir(res_dir)

n_comp = args.n_comp

# File containing the prefixes of the data files in data_dir.
with open(data_dir + '/cluster_vectors/nbrHashes.txt', 'r') as f:
    nb_nzhkmer = int(f.read().splitlines()[0])

with open(data_dir + '/job/sampleList.txt', 'r') as f:
    names_list = f.read().splitlines()
#names_list = names_list[:5]

n_bio_samples = len(names_list)

# Get nonzero hashed k-mers indices from file (obtained with count_nz_hk function)
printt("Get nonzero hashed k-mers indices from file")
print(nb_nzhkmer)

# Load matrix in full format (if it fits in RAM)
printt("Load data matrix")


'''
#use sparse matrix (is not accepted by modl)
import scipy.sparse as spr
X = spr.csc_matrix((nb_nzhkmer,0), dtype = np.float64)
print(X)
for (row_idx, name) in enumerate(names_list):
    nzi_f = data_dir + 'hashed_reads/'+  name + '.count.hash.nzi'
    cond_f = data_dir + 'hashed_reads/'+  name + '.count.hash.cond'
    idxs = np.fromfile(nzi_f, dtype=np.uint64, sep='')
    vals = np.fromfile(cond_f, dtype=np.float32, sep='')
    tV = spr.csc_matrix((vals, (idxs, np.zeros(idxs.size))), shape=(nb_nzhkmer, 1))
    X=spr.hstack((X,tV))
    print(row_idx,X.shape,X.size)
'''

X = np.zeros((nb_nzhkmer, n_bio_samples), dtype=np.float32)
for (row_idx, name) in enumerate(names_list):
    nzi_f = data_dir + 'hashed_reads/'+  name + '.count.hash.nzi'
    cond_f = data_dir + 'hashed_reads/'+  name + '.count.hash.cond'
    idxs = np.fromfile(nzi_f, dtype=np.uint64, sep='')
    vals = np.fromfile(cond_f, dtype=np.float32, sep='')
    X[idxs, row_idx] = vals
    print(row_idx)


print('X.shape',X.shape)
#
# MATRIX FACTORIZATION #
#
# Soft sparsity parameters
code_l1_ratio = args.sps_kmers
comp_l1_ratio = args.sps_egen
# Enforcing the factorization to be non-negative
code_pos = True
comp_pos = True
# Number of eigen-genomes
n_components = n_comp
# Other rguments for the DictFact method from modl
reduction = 1
code_alpha = 1
batch_size = 2000
n_epochs = 4
verbose = 4
n_threads = 4

DF = DictFact(reduction=reduction,
                         code_alpha=code_alpha,
                         code_l1_ratio=code_l1_ratio,
                         comp_l1_ratio=comp_l1_ratio,
                         code_pos=code_pos,
                         comp_pos=comp_pos,
                         n_epochs=n_epochs,
                         n_components=n_components,
                         batch_size=batch_size,
                         verbose=verbose,
                         n_threads=n_threads)

# Run the fit method
printt("Run DF.DictFact method from modl")
DF.fit(X)

# Save output results
components_f = "%s/components_nmf_test0.npy" % (res_dir)
code_f = "%s/code_nmf_test0.npy" % (res_dir)
msg = "Save numpy ndarrays to files\n %s and\n %s\n.\
      Beware that the hkmers indices in the factorized matrix are not the \
      original ones but only the non-zero ones \. Check code if needed" % (components_f, code_f)
printt(msg)

np.save(components_f, DF.components_)
np.save(code_f, DF.code_)
print('DF.components_',DF.components_.shape)
print(DF.components_[:5,:])
print('DF.code_', DF.code_.shape)
print(DF.code_[:5,:])

# Convert results to sparse format (same as input)
# Save matrix ``eigen-genome x hashed-kmer''
printt("Save matrix ``eigen-genome x hashed-kmer'' to files"
       " in sparse coordinate like format ")
for icomp in range(n_comp):
    compgen = DF.code_[:, icomp]
    (_, nzk, nzv) = find(compgen)
    compname = "component%d" % (icomp)
    nzi_f = "%s/%s.count.hash.nzi" % (res_dir, compname)
    vals_f = "%s/%s.count.hash.cond" % (res_dir, compname)
    nzk.tofile(nzi_f, sep='')
    nzv.tofile(vals_f, sep='')
    printt("%d"%(icomp))

# Save matrix ``sample x eigen-genome''
printt("Save matrix ``sample x eigen-genome'' to files"
       " in sparse coordinate like format ")
for (isample, sampname) in enumerate(names_list):
    compsamp = DF.components_[:, isample]
    (_, nzk, nzv) = find(compsamp)
    nzi_f = "%s/%s.count.factor.nzi" % (res_dir, sampname)
    vals_f = "%s/%s.count.factor.cond" % (res_dir, sampname)
    nzk.tofile(nzi_f, sep='')
    nzv.tofile(vals_f, sep='')

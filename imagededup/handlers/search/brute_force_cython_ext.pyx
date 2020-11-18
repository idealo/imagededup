#cython: boundscheck=False, wraparound=False
import cython
from libcpp.vector cimport vector
from libcpp.string cimport string


ctypedef unsigned long long ull

cdef extern from 'builtin/builtin.h':
    int psnip_builtin_popcountll(unsigned long long) nogil

def query(vector[string] all_filenames,
          vector[ull] all_hashes,
          ull query_hash_val,
          unsigned int maxdist):
    matches = []
    cdef unsigned int dist
    cdef string filename
    cdef ull hash_val

    cdef ull i

    for i in range(all_hashes.size()):
        hash_val = all_hashes[i]
        dist = psnip_builtin_popcountll(hash_val ^ query_hash_val)  # requires hash_val and query_hash_val to be integers
        if dist <= maxdist:
            filename = all_filenames[i]
            matches.append((filename.decode('utf-8'), dist))

    return matches
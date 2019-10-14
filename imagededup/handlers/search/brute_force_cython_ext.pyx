#cython: boundscheck=False, wraparound=False
import cython
from libcpp.vector cimport vector
from libcpp.string cimport string


ctypedef unsigned long long ull

cdef extern int __builtin_popcountll(unsigned long long) nogil

cdef vector[string] all_filenames
cdef vector[ull] all_phashes


def add(ull phash, string filename):
    all_phashes.push_back(phash)
    all_filenames.push_back(filename)


def query(ull query_phash, unsigned int maxdist):
    matches = []
    cdef unsigned int dist
    cdef string filename
    cdef ull phash

    cdef ull i

    for i in range(all_phashes.size()):
        phash = all_phashes[i]
        filename = all_filenames[i]
        dist = __builtin_popcountll(phash ^ query_phash)
        if dist <= maxdist:
            matches.append((filename.decode("utf-8"), dist))

    return matches


def clear():
    all_phashes.clear()
    all_filenames.clear()


def size():
    return all_phashes.size()

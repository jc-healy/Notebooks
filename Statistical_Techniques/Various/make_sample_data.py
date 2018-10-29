import numpy as np

def make_missing(A, pct_missing):
    indices = list(np.ndindex(A.shape))
    np.random.shuffle(indices)
    num_to_keep = int(round(pct_missing*len(indices)))
    make_missing = indices[:num_to_keep ]
    for m in make_missing:
        A[m] = np.nan
    return A

def generate_sample_data_cts(n, p, k, pct_missing, noise_scale):
    if p == 0:
        return None
    X = np.random.random([n, k])
    Y = np.random.random([k, p])
    A = np.dot(X,  Y)
    A += np.random.normal(scale = noise_scale, size = A.shape) 
    return make_missing(A, pct_missing)

def generate_sample_data_bin(n, p, k, pct_missing, noise_scale):
    if p == 0:
        return None
    X = np.random.normal(size=[n, k])
    Y = np.random.normal(size=[k, p])
    Atemp = np.dot(X,  Y)
    Atemp += np.random.normal(scale = noise_scale, size = Atemp.shape)
    Abin = (Atemp >0).astype(int).astype(float)
    return make_missing(Abin, pct_missing)

def generate_sample_data_cat(n, p, k, num_cats, pct_missing, noise_scale):
    if p == 0:
        return None
    X = np.random.normal(size=[n, k])
    Y = np.random.normal(size=[k, p]) #we will one-hot the data so this is the the true dimension
    Atemp = np.dot(X,  Y)
    Atemp += np.random.normal(scale = noise_scale, size = Atemp.shape)

    cat_cuts = [-1000] + list(np.percentile(Atemp, [100/num_cats*i for i in range(1,num_cats)])) + [1000]

    Acat = np.zeros_like(Atemp)
    for i in range(len(cat_cuts)-1):
        Acat[(Atemp >= cat_cuts[i]) &  (Atemp < cat_cuts[i+1])] = i+1

    return make_missing(Acat, pct_missing)

def generate_sample_data(n, p_cts, p_bin, p_cat, p_ord, k, num_cats, num_ord, pct_missing, noise_scale):
    k_cts, k_bin, k_cat, k_ord = np.ceil([p_cts, p_bin, p_cat, p_ord] / np.sum([p_cts, p_bin, p_cat, p_ord]) * k).astype(int)
    A_cts = generate_sample_data_cts(n, p_cts, k_cts, pct_missing, noise_scale)
    A_bin = generate_sample_data_bin(n, p_bin, k_bin, pct_missing, noise_scale)
    A_cat = generate_sample_data_cat(n, p_cat, k_cat, num_cats, pct_missing, noise_scale)
    A_ord = generate_sample_data_cat(n, p_ord, k_ord, num_ord, pct_missing, noise_scale)
    A = np.hstack([A for A in [A_cts, A_bin, A_cat, A_ord] if A is not None])
    data_types = ["numerical"]*p_cts +["binary"]*p_bin + ["categorical"]*p_cat + ["ordinal"]*p_ord
    return A, data_types


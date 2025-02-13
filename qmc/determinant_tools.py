import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple

 
def jax_single_occupation_list(mf, weight=1.0):
    '''
    Jax version of single occupation list function.
    Note: This function is just for experimental version
    Original function is in qmc/determinants_from_pyscf and single_determinant_from_mf
    '''
    
    try:
        mf = mf.to_uhf()
    except TypeError:
        mf = mf.to_uhf(mf)
        
    if hasattr(mf, "kpts"):
        occupation = [[jnp.array(jnp.where(k > 0.5)[0]) for k in s] for s in mf.mo_occ] 
    
    else:
        occupation = [jnp.array(jnp.where(s > 0.5)[0]) for s in mf.mo_occ]
        
    return [(weight, occupation)]

def jax_organize_determinant_data(determinant_list, weight_threshold=0):
    '''
    Jax version of organize determinant data function.
    Note: This function is just for experimental version
    create package tool at pyqmc
    
       Input: determinant_list in format 
   [(weight1, ([up_orbs1], [down_orbs1])), 
    (weight2, ([up_orbs2], [down_orbs2])), ...]
   Example: [(0.9, ([0,1], [0,1])), (0.1, ([0,2], [0,1]))]
   
   Returns:
   1. detwt (determinant weights): Coefficient/weight for each determinant
      - Ex: array([0.9, 0.1])  
      - Meaning: First det has weight 0.9, second has 0.1
   
   2. occup (orbital occupations): Unique orbital occupation patterns for each spin
      - Format: [up_patterns, down_patterns]
      - Ex: [[array([0,1]), array([0,2])],  # Up spin occupation patterns
            [array([0,1])]]                 # Down spin occupation patterns
      - Meaning: Stores only unique occupation patterns without duplicates
   
   3. map_dets (pattern mapping): Maps each determinant to its occupation patterns
      - Format: array([[up_indices], [down_indices]])
      - Ex: array([[0,1],   # First det uses up[0], second uses up[1]
                   [0,0]])  # Both dets use down[0]
      - Meaning: Shows which patterns from occup are used by each weight in detwt
    '''
    
    # Initialize empty containers
    weights = []
    patterns = [[], []]
    mapping = [[], []]
    
    filtered_dets = [det for det in determinant_list 
                     if jnp.abs(det[0]) > weight_threshold]
    
    for det in filtered_dets:
        weights.append(det[0])
        spin_occupations = det[1]
        
        for spin in [0, 1]:
            curr_occupation = tuple(spin_occupations[spin])
            
            if curr_occupation not in patterns[spin]:
                mapping[spin].append(len(patterns[spin]))
                patterns[spin].append(curr_occupation)
                
            else:
                pattern_idx = patterns[spin].index(curr_occupation)
                mapping[spin].append(pattern_idx)
                
    determinant_weights = jnp.array(weights)
    pattern_mapping = jnp.array(mapping)
    
    orbital_patterns = [
        [jnp.array(list(pattern)) for pattern in spin_patterns]
        for spin_patterns in patterns
    ]
    
    return determinant_weights, orbital_patterns, pattern_mapping
    
def jax_determinants_from_pyscf(mol, mf, mc = None, tol = -1):
    periodic = hasattr(mol, "a")
    if mc is None:
        determinants = jax_single_occupation_list(mf)
    # elif periodic:
    #     determinants = pbc_determinants_from_casci(mc, cutoff=tol)

    # if mc is not None and not periodic:
    #    determinants = interpret_ci(mc, tol)
    return determinants

def single_occupation_list(mf, weight=1.0):
    try:
        mf = mf.to_uhf()
    except TypeError:
        mf = mf.to_uhf(mf)
    if hasattr(mf, "kpts"):
        occupation = [[list(np.nonzero(k > 0.5)[0]) for k in s] for s in mf.mo_occ]
    else:
        occupation = [list(np.nonzero(s > 0.5)[0]) for s in mf.mo_occ]
        
    return [(weight, occupation)]

def organize_determinant_data(determinant_list, weight_threshold=0):
   """
   Organize determinant information into efficient data structures.

   Args:
       determinant_list: List of tuples in format [(weight, occupation)]
                        where occupation is [[up_spin_orbs], [down_spin_orbs]]
                        Example: [(1.0, [[0,1,2], [0,1]])]
       weight_threshold: Minimum weight to include determinant

   Returns:
       determinant_weights: Array of determinant weights
       orbital_patterns: List of unique orbital occupation patterns [up_patterns, down_patterns] 
       pattern_mapping: Array showing which pattern each determinant uses
   """
   # Initialize empty containers  
   determinant_weights = []
   pattern_mapping = [[], []]  # For up and down spins
   orbital_patterns = [[], []]  # For up and down spins

   # Process each determinant
   for det in determinant_list:
       # Only include if weight is significant
       if np.abs(det[0]) > weight_threshold:
           determinant_weights.append(det[0])
           spin_occupations = det[1]

           # Process each spin channel (0=up, 1=down)
           for spin in [0, 1]:
               # If this is a new orbital pattern
               if spin_occupations[spin] not in orbital_patterns[spin]:
                   pattern_mapping[spin].append(len(orbital_patterns[spin]))
                   orbital_patterns[spin].append(spin_occupations[spin])
               else:
                   # Map to existing pattern
                   pattern_mapping[spin].append(
                       orbital_patterns[spin].index(spin_occupations[spin])
                   )

   return np.array(determinant_weights), orbital_patterns, np.array(pattern_mapping)

def binary_to_occ(S, ncore):
    """
    Converts the binary cistring for a given determinant
    to occupation values for molecular orbitals within
    the determinant.
    """
    occup = [int(i) for i in range(ncore)]
    occup += [int(i + ncore) for i, c in enumerate(reversed(S)) if c == "1"]
    max_orb = max(occup) if occup else 1
    return (occup, max_orb)


def reformat_binary_dets(deters, ncore=0, tol=0):
    #f = lambda x: (binary_to_occ(x[1], ncore)[0], binary_to_occ(x[2], ncore)[0])
    def f(x):
        return (binary_to_occ(x[1], ncore)[0], binary_to_occ(x[2], ncore)[0])
    return [(x[0], f(x)) for x in deters if np.abs(x[0]) > tol]


def create_packed_objects(deters, tol=0):
    """
    if format == "binary":
    deters is expected to be an iterable of tuples, each of which is
    (weight, occupation string up, occupation_string down)
    if format == "list"
    (weight, occupation)
    where occupation is a nested list [s][0, 1, 2, 3 ..], for example.

    ncore should be the number of core orbitals not included in the occupation strings.
    tol is the threshold at which to include the determinants

    :returns:
        * detwt: array of weights for each determinant
        * occup: which orbitals go in which determinants
        * map_dets: given a determinant in detwt, which determinant in occup it corresponds to
    """
    # Create map and occupation objects
    detwt = []
    map_dets = [[], []]
    occup = [[], []]
    for x in deters:
        if np.abs(x[0]) > tol:
            detwt.append(x[0])
            spin_occ = x[1]
            for s in [0, 1]:
                if spin_occ[s] not in occup[s]:
                    map_dets[s].append(len(occup[s]))
                    occup[s].append(spin_occ[s])
                else:
                    map_dets[s].append(occup[s].index(spin_occ[s]))

    return np.array(detwt), occup, np.array(map_dets)

def flatten_determinants(determinants, max_orb, kinds):
    """
    The determinant indices are flattened so that the indices refer to the concatenated MO coefficients.
    """
    determinants_flat = []
    orb_offsets = np.cumsum(max_orb[:, kinds], axis=1)
    orb_offsets = np.pad(orb_offsets[:, :-1], ((0, 0), (1, 0)))
    for wt, det in determinants:
        flattened_det = []
        for det_s, offset_s in zip(det, orb_offsets):
            detlist = [det_s[k] + offset_s[ki] for ki, k in enumerate(kinds)]
            flattened_det.append(list(np.concatenate(detlist).flatten().astype(int)))
        determinants_flat.append((wt, flattened_det))
    return determinants_flat
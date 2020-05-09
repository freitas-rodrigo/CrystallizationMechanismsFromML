from numpy import *
from scipy.special import sph_harm
from scipy.linalg import norm
from ovito.data import *

################################################################################
# Compute fraction of crystal-like bonds (alpha order parameter).              #
################################################################################
# l = spherical harmonics order.
# r_cut = radial cutoff for neighbor finder.
# qq_cut = cutoff for product of Steinhardt order parameter vectors.

def compute_alpha(data, l, r_cut, qq_cut):
  natoms = data.particles.count # Total number of atoms.

  # Compute total number of neighbors of each atom.
  finder = CutoffNeighborFinder(cutoff=r_cut, data_collection=data)
  N_neigh = zeros(natoms, dtype=int)
  for iatom in range(natoms):
    for neigh in finder.find(iatom): 
      N_neigh[iatom] += 1
  N_neigh_total = sum(N_neigh)

  # Unroll neighbor distances.
  r_ij = zeros((N_neigh_total,3)) # Distance vectors to neighbors.
  d_ij = zeros(N_neigh_total) # Distance to neighbors.
  neigh_list = zeros(N_neigh_total,dtype=int) # ID of neighbors.
  ineigh = 0 # Neighbor counter.
  for iatom in range(natoms):
    for neigh in finder.find(iatom):
      r_ij[ineigh] = array(neigh.delta)
      d_ij[ineigh] = neigh.distance
      neigh_list[ineigh] = neigh.index
      ineigh += 1

  # Compute spherical harmonics for all neighbors of each atom.
  phi = arctan2(r_ij[:,1],r_ij[:,0])
  theta = arccos(r_ij[:,2]/d_ij)
  Y = zeros((2*l+1,N_neigh_total),dtype=complex)
  for m in range(-l,l+1):
    Y[m+l] = sph_harm(m,l,phi,theta)

  # Construct Steinhard vector by summing spherical harmonics.
  q = zeros((natoms,2*l+1),dtype=complex)
  ineigh = 0
  for iatom in range(natoms):
    q[iatom] = sum(Y[:,ineigh:ineigh+N_neigh[iatom]],axis=1)
    ineigh += N_neigh[iatom]

  # Normalization.
  for m in range(2*l+1):
    q[:,m] /= N_neigh
  q = (q.T / norm(q,axis=1)).T

  # Classify bonds as crystal-like or not and compute alpha.
  alpha = zeros(natoms) # Crystal-like fraction of bonds per atom.
  ineigh = 0
  for iatom in range(natoms):
    for jatom in neigh_list[ineigh:ineigh+N_neigh[iatom]]:
      qq = dot(q[iatom],conjugate(q[jatom])).real # Im[qq] always 0.
      if qq > qq_cut: alpha[iatom] += 1.0
      ineigh += 1
  alpha /= N_neigh

  return alpha

################################################################################
# Compute radial symmetry functions.                                           #
################################################################################

# This set of functions computes the radial symmetry functions G as defined in in the Nat. Physics paper by Kaxiras and Liu. The rsf are computed only for atoms with ID in ID_compute. The function computes the rsf of one atom at the time, generating a very small memory footprint.

# Radial symmetry function.
def G(x, mu, sigma):
  return exp(-(x-mu)**2/(2*sigma**2))

# Compute radial symmetry functions.
def compute_rsf(data, ID_compute, r_cut, mu, sigma):
  # Setup useful variables.
  n_compute = len(ID_compute) # Number of atoms to compute rsf.
  N_rsf = len(mu) # Number of rsf parameters.
  rsf = zeros((n_compute,N_rsf)) # rsf function for each atom.

  # Find index of all partices in ID_compute.
  ID = array(data.particles['Particle Identifier'])
  index_compute = where(isin(ID,ID_compute,assume_unique=True))[0]

  # Loop over atoms to compute rsf.
  finder = CutoffNeighborFinder(cutoff=r_cut, data_collection=data)
  for iatom in range(n_compute):
    # Compute total number of neighbors.
    N_neigh = 0
    for neigh in finder.find(index_compute[iatom]):
      N_neigh += 1

    # Compute neighbor distances.
    d_ij = zeros(N_neigh,dtype=float) # Distance to neighbors.
    ineigh = 0 # Neighbor counter.
    for neigh in finder.find(index_compute[iatom]):
      d_ij[ineigh] = neigh.distance
      ineigh += 1

    # Compute RSF of iatom atom.
    for i in range(N_rsf):
      rsf[iatom,i] = sum(G(d_ij, mu[i], sigma))

  return rsf

################################################################################

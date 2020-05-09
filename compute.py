from numpy import *

from ovito.io import *
from ovito.modifiers import *

from auxiliary_functions import compute_alpha, compute_rsf

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

################################################################################
# Input parameters and setup.                                                  #
################################################################################

# Parameters for calculation of alpha.
l = 6 # Spherical harmonics order for Steinhard order parameter.
r_cut = 3.0  # rdf cutoff parameter [A].
qq_cut = 0.8 # qq cutoff parameter.

# Parameters for calculation of the RSF.
dmu = 0.40 # rsf Gaussian-mean step [A].
mu = arange(2,10+dmu,dmu) # rsf Gaussian mean [A].
sigma = 0.5 # rsf Gaussian std [A].
rsf_cut = max(mu) + 2*sigma

################################################################################
# Compute alpha at each timestep.                                              #
################################################################################

# Compute alpha.
pipeline = import_file('data/coordinates.dump')
data = pipeline.compute()
#alpha = compute_alpha(data, l=l, r_cut=r_cut, qq_cut=qq_cut)

# Compute rsf.
#ID_compute = arange(1,data.particles.count+1)
#X = compute_rsf(data, ID_compute, r_cut=rsf_cut, mu=mu, sigma=sigma)

## Save data.
#savetxt('data/alpha.dat', alpha, fmt='%.8f')
#savetxt('data/rsf.dat', X, fmt='%.8f')

#X = loadtxt('data/rsf.dat')
alpha = loadtxt('data/alpha.dat')
S = loadtxt('data/alpha.dat')

## Scale data.
scaler = StandardScaler()
#scaler.get_param()
#X = scaler.transform(X)
#
## Compute softness.
#clf = LinearSVC(C=1.0,max_iter=5000)
#clf.get_param()
#S = clf.decision_function(X)
#
## Add softness and alpha to list of particle properties.

# Ovito modifier to add new quantities to the list of particle properties.
def add_prop(frame,data):
  data.particles_.create_property('alpha',data=alpha)
  data.particles_.create_property('S',data=S)
pipeline.modifiers.append(add_prop)
data = pipeline.compute()

# Save dump file with coordinates, alpha, and softness.
export_file(pipeline, 'data/coordinates_and_softness.dump', 'lammps/dump', columns=["Particle Identifier","Particle Type", "Position.X", "Position.Y",     "Position.Z", "alpha", "S"])

################################################################################

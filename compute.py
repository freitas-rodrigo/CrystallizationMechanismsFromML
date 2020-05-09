from numpy import *
from ovito.io import import_file, export_file
from auxiliary_functions import compute_alpha, compute_rsf
import joblib

################################################################################
# Input parameters and setup.                                                  #
################################################################################

# Parameters for calculation of alpha.
l = 6 # Spherical harmonics order for Steinhard parameter.
r_cut = 3.0  # Radial cutoff parameter for neighbor finding.
qq_cut = 0.8 # qq cutoff parameter.

# Parameters for calculation of the radial structure functions.
mu = arange(2,10+0.4,0.4) # rsf Gaussian mean [A].
sigma = 0.5 # rsf Gaussian std [A].
rsf_cut = max(mu) + 2*sigma # Radial cutoff parameter for neighbor finding.

################################################################################
# Compute alpha and rsf for the MD snapshot.                                   #
################################################################################

# Compute alpha.
pipeline = import_file('input_data/coordinates.dump')
data = pipeline.compute()
alpha = compute_alpha(data, l=l, r_cut=r_cut, qq_cut=qq_cut)

# Compute radial structure function.
ID_compute = arange(1,data.particles.count+1)
X = compute_rsf(data, ID_compute, r_cut=rsf_cut, mu=mu, sigma=sigma)

# Scale data.
scaler = joblib.load('input_data/scaler.joblib')
X = scaler.transform(X)

# Compute softness.
clf = joblib.load('input_data/SVM.joblib')
S = clf.decision_function(X)

# Add softness and alpha to list of particle properties.
data.particles_.create_property('alpha',data=alpha)
data.particles_.create_property('softness',data=S)

# Save dump file with coordinates, alpha, and softness.
export_file(data, 'output_data/coordinates_and_softness.dump', 'lammps/dump', columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "alpha", "softness"])

################################################################################

import pystroke
import jax.numpy as jnp
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import numpy as np

#Load GWTC-3 samples
event_GMMs = pystroke.generate_GMMs(
    '/home/jacob.golomb/O3b/nov24/o1o2o3_default/init/result/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json',
    keys=['mass_1'])

pdet_GMM = pystroke.generate_pdet_GMM(
    '/home/rp.o4/offline-injections/mixtures/T2400110-v2/rpo1234-cartesian_spins-semianalytic_o1_o2_o4a-real_o3.hdf',
    '/home/jacob.golomb/O3b/nov24/o1o2o3_default/init/result/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json', 
    keys=['mass_1'])                                   

pistroke_pdet = pystroke.PiStroke(event_GMMs, pdet_GMM)

_ = pistroke_pdet.gradient_descent(iterations=2, delta_diff=-5, tol=1e-2, lower=jnp.array([2]), upper=jnp.array([100]))

np.savetxt('primary_mass_pistroke.dat', pistroke_pdet.result_array_gd)

plt.scatter(pistroke_pdet.result_array_gd[:,0], pistroke_pdet.result_array_gd[:,1])
plt.savefig('primary_mass_test.pdf',bbox_inches='tight')

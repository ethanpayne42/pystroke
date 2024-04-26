import pystroke
import jax.numpy as jnp
import numpyro.distributions as dist
import matplotlib.pyplot as plt

#Load GWTC-3 samples
event_GMMs = pystroke.generate_GMMs('/home/jacob.golomb/O3b/nov24/o1o2o3_default/init/result/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')

pdet_GMM = pystroke.generate_pdet_GMM(
    '/home/reed.essick/rates+pop/o1+o2+o3-sensitivity-estimates/LIGO-T2100377-v2/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5',
    '/home/jacob.golomb/O3b/nov24/o1o2o3_default/init/result/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')                                         

pistroke_pdet = pystroke.PiStroke(event_GMMs, pdet_GMM)

_ = pistroke_pdet.gradient_descent(iterations=2, tol=1e-1)

plt.scatter(pistroke_pdet.result_array_gd[:,0], pistroke_pdet.result_array_gd[:,1])
plt.savefig('test.pdf',bbox_inches='tight')

print(pistroke_pdet.result_array_gd)
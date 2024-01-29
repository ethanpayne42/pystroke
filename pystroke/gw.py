import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import os

from astropy import units
from astropy.cosmology import Planck15

import bilby
import gwpopulation
from gwpopulation_pipe.data_collection import load_all_events

from .distributions import GMMDistribution


# TODO use file:
# /home/jacob.golomb/O3b/nov24/o1o2o3_default/init/result/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json

# Generate prior distribution on chi_eff
qs = bilby.gw.prior.UniformInComponentsMassRatio(0.05,1).sample(10000)
a1s = np.random.uniform(0,1)
a2s = np.random.uniform(0,1)
costilt1s = np.random.uniform(-1,1)
costilt2s = np.random.uniform(-1,1)
chi_eff_prior = gaussian_kde((a1s*costilt1s + qs * a2s * costilt2s)/(1+qs))

def euclidean_distance_prior(redshift):
    luminosity_distance = Planck15.luminosity_distance(redshift).to(units.Gpc).value
    return luminosity_distance**2 * (
        luminosity_distance / (1 + redshift)
        + (1 + redshift)
        * Planck15.hubble_distance.to(units.Gpc).value
        / Planck15.efunc(redshift)
    )

zs_ = np.linspace(0,2.5,1000)
p_z = euclidean_distance_prior(zs_)
p_z /= np.trapz(p_z, zs_)
z_prior = interp1d(zs_, p_z)

def generate_O3_GMMs(population_file, keys=['mass_1'], sample_size=5000):
    
    # Load the population_file
    hyperpe_result = bilby.core.result.read_in_result(population_file)
    hyper_posterior = hyperpe_result.posterior
    hyper_posterior_sample = hyper_posterior.iloc[np.argmax(hyper_posterior["log_likelihood"] + hyper_posterior["log_prior"])]
    sample_dict = {key:hyper_posterior_sample[key] for key in hyper_posterior_sample.keys()}

    sample_dict,_ = gwpopulation.conversions.convert_to_beta_parameters(sample_dict)

    # Load all the relevant events from O3
    args = argparse.Namespace()
    args.mass_prior = 'flat-detector'
    args.spin_prior = 'component'
    args.distance_prior = 'euclidean'
    args.parameters = ['mass_1', 'mass_ratio', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'redshift']
    args.ignore = ['GW170817_GWTC-1', 'S190425z', 'S200105ae', 'S200115j', 'S190426c', 'S190814bv', 'S190917u']
    args.gwtc1_samples_regex = '/home/ethan.payne/projects/evidencemaximizedprior/observing_run_PE/O1O2/GW*.hdf5'
    args.o3a_samples_regex = '/home/ethan.payne/projects/evidencemaximizedprior/observing_run_PE/O3a/*.h5'
    args.o3b_samples_regex = '/home/ethan.payne/projects/evidencemaximizedprior/observing_run_PE/O3b/*.h5'
    args.preferred_labels = ['C01:Mixed', 'Mixed', 'PrecessingSpinIMRHM', 'Overall']
    args.run_dir = 'outdir/'
    args.max_redshift = 1.9
    
    if not os.path.exists(f'{args.run_dir}/data/'):
        os.makedirs(f'{args.run_dir}/data/')

    event_posteriors = load_all_events(args)
    
    event_GMMs = []
    prior_cdfs = []
    
    # Looping over all the events to be considered: 
    for event_str in tqdm(event_posteriors):
        event_name = event_str.split('/')[-1].split('.')[0]
        if event_name not in args.ignore:
            event_posterior = event_posteriors[event_str]

            event_posterior['chi_eff'] = (event_posterior['a_1']*event_posterior['cos_tilt_1'] + event_posterior['mass_ratio']*event_posterior['a_2']*event_posterior['cos_tilt_2']) / \
                (1 + event_posterior['mass_ratio'])

            # Get the individual parameter contributions to the population
            population_prior_dict = {}
            
            mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()

            population_prior_dict['mass_1'] = \
                mass_model.p_m1(pd.DataFrame({'mass_1':event_posterior['mass_1']}),
                    alpha=sample_dict['alpha'],
                    mmin=sample_dict['mmin'], mmax=sample_dict['mmax'],
                    lam=sample_dict['lam'], mpp=sample_dict['mpp'],
                    sigpp=sample_dict['sigpp'], delta_m=sample_dict['delta_m']).values.T
                
            population_prior_dict['q'] = \
                mass_model.p_q(pd.DataFrame({'mass_1':event_posterior['mass_1'], 
                                            'mass_ratio':event_posterior['mass_ratio']}),
                    beta=sample_dict['beta'],
                    mmin=sample_dict['mmin'], delta_m=sample_dict['delta_m'])
            
            population_prior_dict['a_1'] = gwpopulation.utils.beta_dist(
                event_posterior['a_1'], sample_dict['alpha_chi'],
                sample_dict['beta_chi'], scale=sample_dict['amax'])
            population_prior_dict['a_2'] = gwpopulation.utils.beta_dist(
                event_posterior['a_2'], sample_dict['alpha_chi'],
                sample_dict['beta_chi'], scale=sample_dict['amax'])
            
            population_prior_dict['cos_tilt'] = (1 - sample_dict['xi_spin']) / 4 + sample_dict['xi_spin'] * \
                gwpopulation.utils.truncnorm(np.arccos(event_posterior['cos_tilt_1']), 1, sample_dict['sigma_spin'], 1, -1) * \
                gwpopulation.utils.truncnorm(np.arccos(event_posterior['cos_tilt_2']), 1, sample_dict['sigma_spin'], 1, -1)

            redshift_model = gwpopulation.models.redshift.PowerLawRedshift()
            population_prior_dict['z'] = redshift_model(
                pd.DataFrame({'redshift':event_posterior['redshift']}), 
                lamb=sample_dict['lamb']).values.T
    
            # Get the individual contributions to the prior
            prior_dict = {}
            
            prior_dict['mass_1'] = np.array((1 + event_posterior['redshift']) * 
                event_posterior['mass_1'])
            prior_dict['q'] = np.array(1 + event_posterior['redshift'])
            prior_dict['a_1'] = 1 * np.ones(len(event_posterior))
            prior_dict['a_2'] = 1 * np.ones(len(event_posterior))
            prior_dict['cos_tilt'] = 1/4 * np.ones(len(event_posterior))

            prior_dict['z'] = np.array(z_prior(event_posterior['redshift']))
            prior_dict['chi_eff'] = np.array(chi_eff_prior(event_posterior['chi_eff']))
            
            # now assign the weights based on the desired calculations 
            # should set up for 1D, 2D, and N-D!
            if 'chi_eff' not in keys:
                keys_for_prior = ['mass_1', 'q', 'z', 'a_1', 'a_2', 'cos_tilt']
                
            else:
                keys_for_prior = ['mass_1', 'q', 'z', 'chi_eff']
                

            keys_for_astro = deepcopy(keys_for_prior)
            for key in keys:
                keys_for_astro.remove(key)
                    
            weights = \
                np.prod(np.array([prior_dict[key] for key in keys_for_astro]),axis=0) /\
                np.prod(np.array([prior_dict[key] for key in keys_for_prior]),axis=0)
                
            samples_reweighted = event_posterior.sample(
                n=sample_size, replace=True, weights=weights)
            
            # Saving the samples
            keys_for_analysis = deepcopy(keys)
            if 'cos_tilt' in keys:
                keys_for_analysis.remove('cos_tilt')
                keys_for_analysis.append('cos_tilt_1')
                keys_for_analysis.append('cos_tilt_2')
            samples_reduced = samples_reweighted[keys_for_analysis]
            
            print(samples_reduced)
            
    
    # Get the population contributions: 
    
    
    

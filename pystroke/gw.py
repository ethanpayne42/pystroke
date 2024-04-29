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

import bilby
import gwpopulation
from gwpopulation_pipe.data_collection import load_all_events
from gwpopulation_pipe.vt_helper import load_injection_data

from .distributions import GMMDistribution, uniform_generator


# TODO use file:
# /home/jacob.golomb/O3b/nov24/o1o2o3_default/init/result/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json

# Generate prior distribution on chi_eff
qs = bilby.gw.prior.UniformInComponentsMassRatio(0.05,1).sample(10000)
a1s = np.random.uniform(0,1, size=10000)
a2s = np.random.uniform(0,1, size=10000)
costilt1s = np.random.uniform(-1,1, size=10000)
costilt2s = np.random.uniform(-1,1, size=10000)
chi_eff_prior = gaussian_kde((a1s*costilt1s + qs * a2s * costilt2s)/(1+qs))

def euclidean_distance_prior(redshift):
    luminosity_distance = Planck15.luminosity_distance(redshift).to(units.Gpc).value
    return luminosity_distance**2 * (
        luminosity_distance / (1 + redshift)
        + (1 + redshift)
        * Planck15.hubble_distance.to(units.Gpc).value
        / Planck15.efunc(redshift)
    )

zs_ = np.linspace(0,2.9,1000)
p_z = euclidean_distance_prior(zs_)
p_z /= np.trapz(p_z, zs_)
z_prior = interp1d(zs_, p_z)

def generate_GMMs(population_file, keys=['mass_1'], sample_size=5000):
    
    # Load the population_file
    hyperpe_result = bilby.core.result.read_in_result(population_file)
    hyper_posterior = hyperpe_result.posterior
    hyper_posterior_sample = hyper_posterior.iloc[np.argmax(hyper_posterior["log_likelihood"] + hyper_posterior["log_prior"])]
    sample_dict = {key:hyper_posterior_sample[key] for key in hyper_posterior_sample.keys()}

    sample_dict,_ = gwpopulation.conversions.convert_to_beta_parameters(sample_dict)

    # Load all the relevant events from O3    
    args = argparse.Namespace()
    args.mass_prior = {'O4a': 'flat-detector-components', 'O3a': 'flat-detector-components', 
                       'O3b': 'flat-detector-components', 'GWTC1': 'flat-detector-components'}
    args.spin_prior = {'O4a': 'component', 'O3a': 'component', 
                       'O3b': 'component', 'GWTC1': 'component'}
    args.distance_prior = {'O4a': 'comoving', 'O3a': 'euclidean', 'O3b': 'euclidean', 'GWTC1': 'euclidean'}
    args.parameters = ['mass_1', 'mass_ratio', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'redshift']
    args.ignore = ['S231123cg-online', 'S230529ay-Exp0', 'S230802aq-EXP1', 'S230830b-online', 'S230810af-online', 
                   'S231020bw-Exp0', 'S231112ag-online', 'GW170817', 'S190425z', 
                   'S200105ae', 'S200115j', 'S190426c', 'S190814bv', 'S190917u']
    args.sample_regex = {'O4a': '/home/jacob.golomb/o4_pe/RP/pe/*.h5', 
                         'O3b': '/home/jacob.golomb/O3b/O3bPE/S*', 
                         'O3a': '/home/jacob.golomb/O3b/O3aPE/S*', 
                         'GWTC1': '/home/jacob.golomb/o4_pe/RP/pe-o1o2/*.h5'}
    args.preferred_labels = ['Mixed', 'PrecessingSpinIMRHM', 'Overall', 'C01:Mixed', 'IMRPhenomXPHM']
    args.run_dir = 'outdir/'
    args.max_redshift = 1.9
    
    if not os.path.exists(f'{args.run_dir}/data/'):
        os.makedirs(f'{args.run_dir}/data/')

    event_posteriors = load_all_events(args)
    
    event_GMMs = []
    
    # Looping over all the events to be considered: 
    counter = 0
    for event_idx, event_str in tqdm(enumerate(event_posteriors)):
        event_name = event_str.split('/')[-1].split('.')[0]
        print(counter, event_name)
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
                
            population_prior_dict['mass_ratio'] = \
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
            population_prior_dict['redshift'] = redshift_model(
                pd.DataFrame({'redshift':event_posterior['redshift']}), 
                lamb=sample_dict['lamb']).values.T
    
            # Get the individual contributions to the prior
            prior_dict = {}
            
            prior_dict['mass_1'] = np.array((1 + event_posterior['redshift']) * 
                event_posterior['mass_1'])
            prior_dict['mass_ratio'] = np.array(1 + event_posterior['redshift'])
            prior_dict['a_1'] = 1 * np.ones(len(event_posterior))
            prior_dict['a_2'] = 1 * np.ones(len(event_posterior))
            prior_dict['cos_tilt'] = 1/4 * np.ones(len(event_posterior))

            prior_dict['redshift'] = np.array(z_prior(event_posterior['redshift']))
            prior_dict['chi_eff'] = np.array(chi_eff_prior(event_posterior['chi_eff'].values))
            
            # now assign the weights based on the desired calculations 
            # should set up for 1D, 2D, and N-D!
            if 'chi_eff' not in keys:
                keys_for_prior = ['mass_1', 'mass_ratio', 'redshift', 'a_1', 'a_2', 'cos_tilt']
                
            else:
                keys_for_prior = ['mass_1', 'mass_ratio', 'redshift', 'chi_eff']
                

            keys_for_astro = deepcopy(keys_for_prior)
            for key in keys:
                keys_for_astro.remove(key)
                    
            weights = \
                np.prod(np.array([population_prior_dict[key] for key in keys_for_astro]),axis=0) /\
                np.prod(np.array([prior_dict[key] for key in keys_for_prior]),axis=0)
                
            weights = np.nan_to_num(weights)
            weights[weights < 0] = 0
            weights[weights == np.inf] = 0
                
            samples_reweighted = event_posterior.sample(
                frac=0.05, replace=False, weights=weights).reset_index(drop=True)
            
            # Saving the samples
            keys_for_analysis = deepcopy(keys)
            if 'cos_tilt' in keys:
                keys_for_analysis.remove('cos_tilt')
                keys_for_analysis.append('cos_tilt_1')
                keys_for_analysis.append('cos_tilt_2')
            samples_reduced = samples_reweighted[keys_for_analysis]
            
            
            prior_cdfs = []
            for key in keys:
                max = np.max(event_posterior[key])*1.01
        
                if np.min(event_posterior[key]) > 0:
                    min = np.min(event_posterior[key])*0.99
                else:
                    min = np.min(event_posterior[key])*1.01
                
                prior_cdfs.append(uniform_generator(min, max))
            
            event_GMM = GMMDistribution(samples_reduced.to_numpy(), prior_cdfs)
            
            event_GMMs.append(event_GMM)
            counter += 1
        
    return event_GMMs


def generate_pdet_GMM(pdet_file, population_file, keys=['mass_1'], snr_threshold=10, ifar_threshold=1, sample_size=10000):
    # pdet file: /home/reed.essick/rates+pop/o1+o2+o3-sensitivity-estimates/LIGO-T2100377-v2/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5
    # pop file: /home/jacob.golomb/O3b/nov24/o1o2o3_default/init/result/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json
    
    # Reading in the injections
    found_injections_data = pd.DataFrame.from_dict(
        load_injection_data(pdet_file, snr_threshold=snr_threshold, ifar_threshold=ifar_threshold).to_dict())
    found_injections_data = found_injections_data.assign(chi_eff=(found_injections_data['a_1']*found_injections_data['cos_tilt_1'] + \
        found_injections_data['mass_ratio'] * found_injections_data['a_2'] * found_injections_data['cos_tilt_2']) /\
        (1+found_injections_data['mass_ratio']))
    
    # Load the population_file
    hyperpe_result = bilby.core.result.read_in_result(population_file)
    hyper_posterior = hyperpe_result.posterior
    hyper_posterior_sample = hyper_posterior.iloc[np.argmax(hyper_posterior["log_likelihood"] + hyper_posterior["log_prior"])]
    sample_dict = {key:hyper_posterior_sample[key] for key in hyper_posterior_sample.keys()}

    sample_dict,_ = gwpopulation.conversions.convert_to_beta_parameters(sample_dict)
    
    # Getting the population_prior_dict
    population_prior_dict = {}
            
    mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()

    population_prior_dict['mass_1'] = \
        mass_model.p_m1(pd.DataFrame({'mass_1':found_injections_data['mass_1']}),
            alpha=sample_dict['alpha'],
            mmin=sample_dict['mmin'], mmax=sample_dict['mmax'],
            lam=sample_dict['lam'], mpp=sample_dict['mpp'],
            sigpp=sample_dict['sigpp'], delta_m=sample_dict['delta_m']).values.T
        
    population_prior_dict['mass_ratio'] = \
        mass_model.p_q(pd.DataFrame({'mass_1':found_injections_data['mass_1'], 
                                    'mass_ratio':found_injections_data['mass_ratio']}),
            beta=sample_dict['beta'],
            mmin=sample_dict['mmin'], delta_m=sample_dict['delta_m'])
    
    population_prior_dict['a_1'] = gwpopulation.utils.beta_dist(
        found_injections_data['a_1'], sample_dict['alpha_chi'],
        sample_dict['beta_chi'], scale=sample_dict['amax'])
    population_prior_dict['a_2'] = gwpopulation.utils.beta_dist(
        found_injections_data['a_2'], sample_dict['alpha_chi'],
        sample_dict['beta_chi'], scale=sample_dict['amax'])
    
    population_prior_dict['cos_tilt'] = (1 - sample_dict['xi_spin']) / 4 + sample_dict['xi_spin'] * \
        gwpopulation.utils.truncnorm(np.arccos(found_injections_data['cos_tilt_1']), 1, sample_dict['sigma_spin'], 1, -1) * \
        gwpopulation.utils.truncnorm(np.arccos(found_injections_data['cos_tilt_2']), 1, sample_dict['sigma_spin'], 1, -1)

    redshift_model = gwpopulation.models.redshift.PowerLawRedshift()
    population_prior_dict['redshift'] = redshift_model(
        pd.DataFrame({'redshift':np.array(found_injections_data['redshift'])}), 
        lamb=sample_dict['lamb']).values.T
    
    # now assign the weights based on the desired calculations 
    # should set up for 1D, 2D, and N-D!
    if 'chi_eff' not in keys:
        keys_for_prior = ['mass_1', 'mass_ratio', 'redshift', 'a_1', 'a_2', 'cos_tilt']
        
    else:
        keys_for_prior = ['mass_1', 'mass_ratio', 'redshift', 'chi_eff']
        

    keys_for_astro = deepcopy(keys_for_prior)
    for key in keys:
        keys_for_astro.remove(key)

    weights = \
        np.prod(np.array([population_prior_dict[key] for key in keys_for_astro]),axis=0) /\
        found_injections_data['prior']
    
    weights = np.nan_to_num(weights)
    weights[weights < 0] = 0
    weights[weights > 1e100] = 0
    
    samples_reweighted = found_injections_data.sample(
        frac=0.05, replace=False, weights=weights).reset_index(drop=True)
    
    # Saving the samples
    keys_for_analysis = deepcopy(keys)
    if 'cos_tilt' in keys:
        keys_for_analysis.remove('cos_tilt')
        keys_for_analysis.append('cos_tilt_1')
        keys_for_analysis.append('cos_tilt_2')
    samples_reduced = samples_reweighted[keys_for_analysis]
    
    prior_cdfs = []
    for key in keys:
        max = np.max(found_injections_data[key])*1.01+0.1
        
        if np.min(found_injections_data[key]) > 0:
            min = np.min(found_injections_data[key])*0.99-0.1
        else:
            min = np.min(found_injections_data[key])*1.01-0.1
        
        prior_cdfs.append(uniform_generator(min, max))
    
    pdet_GMM = GMMDistribution(samples_reduced.to_numpy(), prior_cdfs)
    
    return pdet_GMM
    
            
    
    
    
    

import jax
import jaxopt
import jax.numpy as jnp
import numpy as np
import scipy
import tqdm

class PiStroke(object):
    
    def __init__(self, observation_list, detection_probability):
        """
        TODO
        """
        self.dimensions = observation_list[0].dimensions
        self.observation_list = observation_list
        self.detection_probability = detection_probability
        self.N_observations = len(self.observation_list)
        
        # Set up the array of means
        self.obs_means = []
        self.obs_vars = []
        for i in range(self.N_observations):
            self.obs_means.append(jnp.mean(self.observation_list[i].samples,axis=0))
            self.obs_vars.append(jnp.var(self.observation_list[i].samples,axis=0))

        # Set up multiple result arrays for different methods
        self.result_array_gd = None
        
    def log_Lstroke(self, pistroke_array):
        """
        TODO
        """
        pdet_term = - self.N_observations * jax.scipy.special.logsumexp(
            self.detection_probability.log_prob(pistroke_array[:,:self.dimensions]).T + pistroke_array[:,-1])
        
        pdet_term_noinfs = jax.lax.select(jnp.isfinite(pdet_term), pdet_term, -jnp.inf)

        data_term = jnp.array(0)
        for i in range(self.N_observations):
            data_term += jax.scipy.special.logsumexp(
                self.observation_list[i].log_prob(pistroke_array[:,:self.dimensions]).T + pistroke_array[:,-1])
        
        log_L = data_term + pdet_term_noinfs
        
        return jnp.nan_to_num(log_L, nan=-jnp.inf)
    
    def negative_log_Lstroke(self, pistroke_array):
        """
        TODO
        """
        logL = self.log_Lstroke(pistroke_array)
        return - logL
        
    def gradient_descent(self, iterations=5, delta_diff=1e-4, tol=1e-4, lower=jnp.array([5]), upper=jnp.array([100])):
        """
        TODO
        """
        if self.result_array_gd is None:
            for i in tqdm.tqdm(range(iterations)):
                key = jax.random.PRNGKey(np.random.randint(0,1000000))
                
                init_position = np.zeros((self.N_observations, self.dimensions))
                
                for obs_idx in range(self.N_observations):
                    init_position[obs_idx] = np.random.multivariate_normal(
                        self.obs_means[obs_idx],
                        np.diag(self.obs_vars[obs_idx]))
                
                init_pistroke = jnp.atleast_2d(jnp.hstack([
                    jnp.array(init_position),
                    -jnp.log(self.N_observations)*jnp.ones((int(self.N_observations),1))]))
                
                print(f'Random initialized position has logL {self.log_Lstroke(init_pistroke)}')
                
                # Initial run 
                opt_obj = jaxopt.ProjectedGradient(self.negative_log_Lstroke, projection=jaxopt.projection.projection_box)
                res = opt_obj.run(
                    init_params=init_pistroke, 
                    hyperparams_proj=(jnp.concatenate([jnp.atleast_1d(lower), jnp.atleast_1d(jnp.array([-jnp.inf]))]), 
                                      jnp.concatenate([jnp.atleast_1d(upper), jnp.atleast_1d(jnp.array([jnp.inf]))])))
                pistroke = res.params
                print(pistroke)
                
                print(f'Minimized result has logL {self.log_Lstroke(pistroke)}')
                
                # TODO don't think I need the reduction step
                pistroke = self.construct_reduced_pistroke(pistroke, delta_diff=delta_diff)
                res = opt_obj.run(
                    init_params=pistroke,
                    hyperparams_proj=(jnp.concatenate([jnp.atleast_1d(lower), jnp.atleast_1d(jnp.array([-jnp.inf]))]), 
                                      jnp.concatenate([jnp.atleast_1d(upper), jnp.atleast_1d(jnp.array([jnp.inf]))])))
                
                pistroke = res.params
                print(pistroke)
                print(f'Minimized and reduced result has logL {self.log_Lstroke(pistroke)}')
                
                # pistroke = self.combine_delta_functions(pistroke, tol=tol)
                # print(pistroke)

                # print(f'Minimized, reduced, and combined result has logL {self.log_Lstroke(pistroke)}')
                
                if i == 0:
                    self.result_array_gd = pistroke
                    maxL = self.log_Lstroke(pistroke)
                    
                else:
                    if self.log_Lstroke(pistroke) > maxL:
                        self.result_array_gd = pistroke
                        maxL = self.log_Lstroke(pistroke)
                        
        print(f'\n Final logL {self.log_Lstroke(self.result_array_gd)}')
        return self.result_array_gd
    
    
    def construct_reduced_pistroke(self, pistroke_array, delta_diff=-4):
        """
        TODO
        """
        
        # Construct sorted pistroke
        sorted_pistroke = jnp.array(pistroke_array[pistroke_array[:, -1].argsort()[::-1]])
        
        # Construct reduced pistroke
        reduced_pistroke = []
        max_weight = jnp.max(sorted_pistroke[:,-1])
        for i in range(len(sorted_pistroke)):
            
            if sorted_pistroke[i, -1] - max_weight > delta_diff:
                reduced_pistroke.append(sorted_pistroke[i])
        
        reduced_pistroke = jnp.array(reduced_pistroke)
        reduced_pistroke = reduced_pistroke.at[:,-1].set(reduced_pistroke[:,-1] - scipy.special.logsumexp(reduced_pistroke[:,-1]))
        
        return jnp.array(reduced_pistroke[reduced_pistroke[:, -1].argsort()[::-1]])
        
    def combine_delta_functions(self, pistroke_array, tol=1e-4):
        """
        TODO
        """
        
        # Construct sorted pistroke
        sorted_pistroke = jnp.array(pistroke_array[pistroke_array[:, -1].argsort()[::-1]])
        
        # Construct reduced pistroke
        reduced_pistroke = []
        for i in range(len(sorted_pistroke)):
            matching_location = False
            matching_index = 0
            
            for j in range(len(reduced_pistroke)):
                if np.linalg.norm((sorted_pistroke[i,:-1] - reduced_pistroke[j][:-1])/tol) < 1:
                    matching_location = True
                    matching_index = j
                    break
                    
            if matching_location:
                reduced_pistroke[matching_index].at[-1].set(jnp.logaddexp(
                    reduced_pistroke[matching_index][-1], sorted_pistroke[i,-1]))
            else:
                reduced_pistroke.append(sorted_pistroke[i])
        
        reduced_pistroke = jnp.array(reduced_pistroke)
        reduced_pistroke = reduced_pistroke.at[:,-1].set(reduced_pistroke[:,-1] - scipy.special.logsumexp(reduced_pistroke[:,-1]))
        
        return jnp.array(reduced_pistroke[reduced_pistroke[:, -1].argsort()[::-1]])
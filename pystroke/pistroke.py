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
        self.dimensions = detection_probability.shape()[0]
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

        data_term = jnp.array(0)
        for i in range(self.N_observations):
            data_term += jax.scipy.special.logsumexp(
                self.observation_list[i].log_prob(pistroke_array[:,:self.dimensions]).T + pistroke_array[:,-1])
        
        return data_term + pdet_term
    
    def negative_log_Lstroke(self, pistroke_array):
        """
        TODO
        """
        return -self.log_Lstroke(pistroke_array)
        
    def gradient_descent(self, iterations=5, tol=1e-4, weight_min_level=1e-6):
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
                opt_obj = jaxopt.GradientDescent(self.negative_log_Lstroke)
                res = opt_obj.run(init_params=init_pistroke)
                pistroke = res.params
                
                print(f'Minimized result has logL {self.log_Lstroke(pistroke)}')
                
                # TODO don't think I need the reduction step
                pistroke = self.construct_reduced_pistroke(pistroke, tol=tol, weight_min_level=weight_min_level)
                res = opt_obj.run(init_params=pistroke)
                pistroke = res.params
                #pistroke = self.construct_reduced_pistroke(pistroke, tol=tol, weight_min_level=weight_min_level)

                print(f'Minimized and reduced result has logL {self.log_Lstroke(pistroke)}')
                
                if i == 0:
                    self.result_array_gd = pistroke
                    maxL = self.log_Lstroke(pistroke)
                    
                else:
                    if self.log_Lstroke(pistroke) > maxL:
                        self.result_array_gd = pistroke
                        maxL = self.log_Lstroke(pistroke)
                        
        print(f'\n Final logL {self.log_Lstroke(self.result_array_gd)}')
        return self.result_array_gd
    
    
    def construct_reduced_pistroke(self, pistroke_array, tol=1e-4, weight_min_level=1e-10):
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
        
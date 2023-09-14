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
                self.observation_list[i].log_prob(pistroke_array[:,:self.dimensions]) + pistroke_array[:,-1])
        
        return pdet_term + data_term
    
    def negative_log_Lstroke(self, pistroke_array):
        """
        TODO
        """
        return -self.log_Lstroke(pistroke_array)
        
    def gradient_descent(self, iterations=5):
        """
        TODO
        """
        if self.result_array_gd is None:
            for i in tqdm.tqdm(range(iterations)):
                key = jax.random.PRNGKey(np.random.randint(0,1000000))
                
                init_pistroke = jnp.atleast_2d(jnp.hstack([
                    jax.random.uniform(key, minval=-3,maxval=3, shape=(self.N_observations, self.dimensions)), 
                    jnp.zeros((int(self.N_observations),1))]))
                
                print(f'Random initialized position has logL {self.log_Lstroke(init_pistroke)}')
                
                # Initial run 
                opt_obj = jaxopt.GradientDescent(self.negative_log_Lstroke)
                res = opt_obj.run(init_params=init_pistroke)
                pistroke = res.params
                
                print(f'Minimized result has logL {self.log_Lstroke(pistroke)}')
                
                pistroke = self.construct_reduced_pistroke(pistroke)
                res = opt_obj.run(init_params=pistroke)
                pistroke = res.params
                
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
    
    
    def construct_reduced_pistroke(self, pistroke_array, tol=1e-2, weight_min_level=1e-5):
        """
        TODO
        """
        
        # Construct sorted pistroke
        sorted_pistroke = np.array(pistroke_array[pistroke_array[:, -1].argsort()[::-1]])
        
        # Construct reduced pistroke
        reduced_pistroke = []
        for i in range(len(sorted_pistroke)):
            matching_location = False
            matching_index = 0
            
            for j in range(len(reduced_pistroke)):
                if np.linalg.norm(sorted_pistroke[i,:-1] - reduced_pistroke[j][:-1]) < tol:
                    matching_location = True
                    matching_index = j
                    break
                    
            if matching_location:
                reduced_pistroke[matching_index][-1] = np.logaddexp(
                    reduced_pistroke[matching_index][-1], sorted_pistroke[i,-1])
            else:
                reduced_pistroke.append(sorted_pistroke[i])
        
        reduced_pistroke = np.array(reduced_pistroke)
        
        reduced_pistroke[:,-1] -= scipy.special.logsumexp(reduced_pistroke[:,-1])
        pistroke_array = reduced_pistroke[reduced_pistroke[:,-1] > np.log(weight_min_level)]
    
        return pistroke_array
        
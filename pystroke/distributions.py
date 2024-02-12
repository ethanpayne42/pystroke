import jax.numpy as jnp
import jax
from sklearn import mixture
import numpy as np

def uniform_generator(minimum, maximum):
    
    def uniform(x):
        arr = (x - minimum)/(maximum - minimum)
        arr = arr.at[arr < 0].set(0)
        arr = arr.at[arr > 1].set(1)

    return uniform

class GMMDistribution(object):
    
    def __init__(self, samples, priors_cdfs):
        self.samples = jnp.array(samples)
        self.dimensions = jnp.shape(self.samples)[1]
        self.priors_cdfs = priors_cdfs
        
        self.transformed_samples = self.transform_samples(self.samples)
        
        self.GMM_fitted_distribution = self.fit_GMM(self.transformed_samples)
        
        # Extract the means and standard deviations
        self.GMM_n_comps = self.GMM_fitted_distribution.n_components
        self.GMM_means = self.GMM_fitted_distribution.means_
        self.GMM_precisions_chol = self.GMM_fitted_distribution.precisions_cholesky_
        
    
    def log_prob(self, points):
        """
        Taken from GMM code
        
        Note this is actually the log likelihood
        """
        
        X = self.transform_samples(points).T
        
        log_norm = - 0.5 * (self.dimensions * jnp.log(2 * np.pi) + jnp.linalg.norm(X, axis=1)**2 + jnp.log(self.dimensions))
        
        log_det = jnp.sum(jnp.log(self.GMM_precisions_chol.reshape(self.GMM_n_comps, -1)[:, :: self.dimensions + 1]), 1)
        
        log_p = jnp.empty((len(points), self.GMM_n_comps))
        for k, (mu, prec_chol) in enumerate(zip(self.GMM_means, self.GMM_precisions_chol)):
            y = jnp.dot(X - mu, prec_chol)
            log_p = log_p.at[:,k].set(jnp.sum(jnp.square(y), axis=1))
        
        # return jax.scipy.special.logsumexp(-0.5 * (self.dimensions * jnp.log(2 * np.pi) + log_p) + log_det + jnp.log(self.GMM_fitted_distribution.weights_), axis=1) - log_norm
        # Removed + log_det  as the normalization is not important here - removed from here    VVVVVV
        return jax.scipy.special.logsumexp(-0.5 * (self.dimensions * jnp.log(2 * np.pi) + log_p) + jnp.log(self.GMM_fitted_distribution.weights_), axis=1) - log_norm
    
    def transform_samples(self, samples):
        
        transformed_samples = []
        
        for i in range(self.dimensions):
           transformed_samples.append(jnp.sqrt(2)*jax.scipy.special.erfinv(2*self.priors_cdfs[i](samples[:,i])-1))
        
        return jnp.array(transformed_samples)
    
    
    def fit_GMM(self, transformed_samples):
        
        best_gmm = mixture.GaussianMixture(n_components=1).fit(transformed_samples[:int(0.8*len(transformed_samples.T))].T)
        best_bic = best_gmm.bic(transformed_samples[:int(len(transformed_samples.T))-int(0.2*len(transformed_samples.T))].T)
        
        for n_comp in range(2,16):
            gmm = mixture.GaussianMixture(n_components=n_comp).fit(transformed_samples[:int(0.8*len(transformed_samples.T))].T)
            bic = gmm.bic(transformed_samples[:int(len(transformed_samples.T))-int(0.2*len(transformed_samples.T))].T)
            
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        
        return best_gmm
    
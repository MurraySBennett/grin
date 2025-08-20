import numpy as np 
from scipy.stats import multivariate_normal
import os
from tqdm import tqdm

class GRTDataGenerator:
    def __init__(self, num_matrices=10, num_dimensions=2, num_levels=2, trial_range=(25, 1000)):
        self.num_matrices = num_matrices
        self.num_dimensions = num_dimensions
        self.num_levels = num_levels
        self.num_stimuli = self.num_dimensions * self.num_levels
        
        self.mean_range = (-1, 1)
        self.corr_range = (-0.99, 0.99) # Avoids singular matrices
        self.var_range = (0.01, 0.5)
        self.crit_range = (self.mean_range[0] * 0.8, self.mean_range[1] * 0.8)
        self.sample_loss = 0.1
        self.trial_range = trial_range

        # in order of constraints
        self.model_names = [
            'pi_ps_ds', # 4
            'rho1_ps_ds', # 5
            'pi_psa_ds', 'pi_psb_ds', # 6
            'rho1_psa_ds', 'rho1_psb_ds', # 7
            'pi_ds', 'ps_ds', # 8
            'rho1_ds', # 9
            'psa_ds', 'psb_ds', # 10
            'ds', # 12
        ]

    def generate_cm(self, model_name, n_stimulus_trials):
        means, cov_mat, c = self.random_model_params(model_name)
        samples = self.get_samples(means, cov_mat, n_stimulus_trials, sample_loss_factor=self.sample_loss)
        cm_rows = []
        trial_counts = np.array([])
        for _, stim_samples in enumerate(samples):
            response_counts = self.get_response_counts(stim_samples, c)
            cm_rows.append(response_counts)
            trial_counts = np.concatenate([trial_counts, [len(stim_samples)]])
        cm = np.vstack(cm_rows)
        
        flat_params = np.concatenate([means.flatten(), np.array([m.flatten() for m in cov_mat]).flatten(), c])
        return cm, trial_counts, flat_params, means, cov_mat, c

    def generate_cms(self, model_name, n_matrices=1):
        cms, trial_counts, params = [], [], []
        for i in tqdm(range(n_matrices), total=n_matrices, desc=f"Generating matrices for {model_name}"):
            n_trials = int(np.round(np.random.uniform(*self.trial_range)))
            cm, trial_count, flat_params, _, _, _ = self.generate_cm(model_name, n_trials)
            cms.append(cm.flatten())
            trial_counts.append(trial_count)
            params.append(flat_params)
        return cms, trial_counts, params

    def generate_all_model_cms(self):
        y_cls = []
        y_cls_label = []
        trial_counts = []
        parameters = []
        cms = []
        for model_class, model_label in enumerate(self.model_names):
            model_cms, model_trial_counts, model_params = self.generate_cms(model_label, self.num_matrices)
            cms.extend(model_cms)
            parameters.extend(model_params)
            trial_counts.extend(model_trial_counts)
            y_cls.extend([model_class] * self.num_matrices)
            y_cls_label.extend([model_label] * self.num_matrices)
        
        return np.array(cms), np.array(parameters), np.array(trial_counts), np.array(y_cls), np.array(y_cls_label)
    
    def random_model_params(self, model_name="ds"):
        means = self.random_means()
        cov_mat = self.get_cov_mats()
        c = self.random_crit()
        
        if 'psa' in model_name:
            means = self._set_psa(means)
        if 'psb' in model_name:
            means = self._set_psb(means)
        if 'ps_ds' == model_name:
            means = self._set_ps(means)
        
        if 'pi' in model_name:
            cov_mat = self._set_pi(cov_mat)
        if 'rho1' in model_name:
            cov_mat = self._set_rho1(cov_mat)
            
        return means, cov_mat, c

    def random_crit(self):
        return np.round(np.random.uniform(*self.crit_range, 2), 2)

    def get_samples(self, means, cov_mat, size, sample_loss_factor=0.1):
        dist_samples = [
            multivariate_normal.rvs(
                mean=means[i*2:i*2+2],
                cov=cov_mat[i],
                size=int(np.random.uniform(size * (1 - sample_loss_factor), size))
            ) for i in range(self.num_stimuli)
        ]
        return dist_samples

    def get_response_counts(self, samples, c):
        counts = np.zeros(4, dtype=int)
        counts[0] = np.sum((samples[:, 0] < c[0]) & (samples[:, 1] < c[1]))
        counts[1] = np.sum((samples[:, 0] < c[0]) & (samples[:, 1] >= c[1]))
        counts[2] = np.sum((samples[:, 0] >= c[0]) & (samples[:, 1] < c[1]))
        counts[3] = np.sum((samples[:, 0] >= c[0]) & (samples[:, 1] >= c[1]))
        return counts

    def random_means(self):
        means = np.zeros(self.num_stimuli * self.num_dimensions, dtype=float)
        means[2:] = np.round(np.random.uniform(*self.mean_range, self.num_stimuli * self.num_dimensions - 2), 2)
        return means

    def random_cov_mat(self):
        # Generate random eigenvalues
        # They must be positive to ensure a positive definite matrix
        eigenvalues = np.random.uniform(self.var_range[0], self.var_range[1], self.num_dimensions)
        # Generate a random orthogonal matrix (Q) for eigenvectors
        # This is a random rotation matrix
        Q, _ = np.linalg.qr(np.random.rand(self.num_dimensions, self.num_dimensions))
        # Construct the covariance matrix using the spectral decomposition
        # Formula: Cov = Q @ Lambda @ Q.T
        # Where Q is the matrix of eigenvectors and Lambda is the diagonal matrix of eigenvalues
        diag_eigenvalues = np.diag(eigenvalues)
        cov_mat = Q @ diag_eigenvalues @ Q.T
        # Ensure the matrix is perfectly symmetric due to potential floating point errors
        cov_mat = (cov_mat + cov_mat.T) / 2
        
        return np.round(cov_mat, 2)
    
    def get_cov_mats(self):
        cov_mat = [self.random_cov_mat() for _ in range(self.num_stimuli)]
        return np.array(cov_mat)
    
    def _set_pi(self, cov_mat):
        for i in range(self.num_stimuli):
            temp_cov = np.diag(np.diag(cov_mat[i]))
            cov_mat[i] = temp_cov
        return cov_mat
    
    def _set_rho1(self, cov_mat):
        rho1 = self.random_cov_mat()
        for i in range(self.num_stimuli):
            cov_mat[i] = rho1.copy()
        return cov_mat

    def _set_ps(self, means):
        means[3] = means[1]
        means[5] = means[7]
        means[4] = means[0]
        means[6] = means[2]
        return means

    def _set_psa(self, means):
        means[3] = means[1]
        means[5] = means[7]
        return means

    def _set_psb(self, means):
        means[4] = means[0]
        means[6] = means[2]
        return means

if __name__ == "__main__":
    gen = GRTDataGenerator(num_matrices=10, num_dimensions=2, num_levels=2, trial_range=(10, 2000))
    cms, parameters, trial_counts, y_cls, y_cls_label = gen.generate_all_model_cms()
    np.savez('demo_grt_dataset.npz', X=cms, X_trials=trial_counts, y_params=parameters, y_model_cls=y_cls, y_cls_label=y_cls_label)
    np.savez(os.path.join(MODELS_DIR, 'data_splits.npz'), 
             train_indices=indices_train, 
             val_indices=indices_val)
    print("Data split indices saved to data_splits.npz")

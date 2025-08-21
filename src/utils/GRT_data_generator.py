import numpy as np 
from scipy.stats import multivariate_normal
import os
from tqdm import tqdm
from src.utils.config import *
import argparse

class GRTDataGenerator:
    def __init__(self, num_matrices=10, num_dimensions=2, num_levels=2, trial_range=TRIALS_RANGE):
        self.num_matrices = num_matrices
        self.num_dimensions = num_dimensions
        self.num_levels = num_levels
        self.num_stimuli = self.num_dimensions * self.num_levels
        
        self.mean_range = (-1, 1)
        self.corr_range = (-0.99, 0.99) # Avoids singular matrices
        self.min_rho1 = 0.05
        self.min_mean_diff = 0.1 # ensuring means differ when they ought to (e.g., when psa present, b dims should be different) - this is kinda large.. too large for practical purposes? Test this stuff later - for now, just testing to see if the network runs.
        self.var_range = (0.1, 1.5)
        self.crit_range = (self.mean_range[0] * 0.8, self.mean_range[1] * 0.8)
        self.coin_flip = 0.8
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
            n_trials = int(np.ceil(np.random.uniform(*self.trial_range)))
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
        
        if 'pi' in model_name:
            cov_mat = self._set_pi(cov_mat)
        if 'rho1' in model_name:
            cov_mat = self._set_rho1(cov_mat)

        if 'psa' in model_name:
            means = self._set_psa(means)
            while (np.abs(means[3] - means[1]) < self.min_mean_diff or
                   np.abs(means[5] - means[7]) < self.min_mean_diff):
                means = self.random_means()
                means = self._set_psa(means)
        if 'psb' in model_name:
            means = self._set_psb(means)
            while (np.abs(means[4] - means[0]) < self.min_mean_diff or
                   np.abs(means[6] - means[2]) < self.min_mean_diff):
                means = self.random_means()
                means = self._set_psb(means)
        if 'ps_ds' == model_name:
            means = self._set_ps(means)
        
        # Ensure 'pi_ds' doesn't accidentally become 'pi_psa_ds' or 'pi_psb_ds'
        if model_name == 'pi_ds':
             while (np.abs(means[4] - means[0]) < self.min_mean_diff and
                    np.abs(means[6] - means[2]) < self.min_mean_diff) or \
                   (np.abs(means[3] - means[1]) < self.min_mean_diff and
                    np.abs(means[5] - means[7]) < self.min_mean_diff):
                means = self.random_means()
        # Ensure 'rho1_ds' doesn't accidentally become 'rho1_psa_ds' or 'rho1_psb_ds'
        if model_name == 'rho1_ds':
             while (np.abs(means[4] - means[0]) < self.min_mean_diff and
                    np.abs(means[6] - means[2]) < self.min_mean_diff) or \
                   (np.abs(means[3] - means[1]) < self.min_mean_diff and
                    np.abs(means[5] - means[7]) < self.min_mean_diff):
                means = self.random_means() 
            
        c = self.random_crit(means)

        return means, cov_mat, c

    def random_crit(self, means):
        means_dim1_range = (min(means[0], means[2]), max(means[0], means[2]))
        means_dim2_range = (min(means[1], means[3]), max(means[1], means[3]))
        if np.random.rand() < self.coin_flip:
            crit1 = np.random.uniform(*means_dim1_range)
            crit2 = np.random.uniform(*means_dim2_range)
            return np.round(np.array([crit1, crit2]), 2)
        else:
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
        samples = np.atleast_2d(samples)
        counts = np.zeros(4, dtype=int)
        counts[0] = np.sum((samples[:, 0] < c[0]) & (samples[:, 1] < c[1]))
        counts[1] = np.sum((samples[:, 0] < c[0]) & (samples[:, 1] >= c[1]))
        counts[2] = np.sum((samples[:, 0] >= c[0]) & (samples[:, 1] < c[1]))
        counts[3] = np.sum((samples[:, 0] >= c[0]) & (samples[:, 1] >= c[1]))
        
        return counts


    def random_means(self):
        means = np.zeros(self.num_stimuli * self.num_dimensions, dtype=float)
        means[2:] = np.random.uniform(*self.mean_range, self.num_stimuli * self.num_dimensions - 2)

        if np.random.rand() < self.coin_flip:
            # Stimulus 1 has a greater x-mean than stimulus 0
            means[2] = np.random.uniform(0.05, self.mean_range[1]) # Ensure s1 is in positive space
        if np.random.rand() < self.coin_flip:
            # stimulus 2 has a greater y-mean than stimulus 0
            means[5] = np.random.uniform(0.05, self.mean_range[1])
        if np.random.rand() < self.coin_flip:
            # stimulus 3 has a greater x-mean than stimulus 2
            means[4], means[6] = np.sort([means[4], means[6]])
        if np.random.rand() < self.coin_flip:
            # stimulus 3 has a greater y-mean than stimulus 1
            means[3], means[7] = np.sort([means[3], means[7]])
        return np.round(means, 3)

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
        new_cov_mat = []
        for _ in range(self.num_stimuli):
            off_diag_noise = np.random.uniform(-0.01, 0.01)
            variances = np.random.uniform(self.var_range[0], self.var_range[1], self.num_dimensions)
            diag_matrix = np.diag(variances)
            diag_matrix[0, 1] = off_diag_noise
            diag_matrix[1, 0] = off_diag_noise
            new_cov_mat.append(diag_matrix)
        return np.array(new_cov_mat)

    def _set_rho1(self, cov_mat):
        rho1_cov_mat = self.random_cov_mat()
        while np.abs(rho1_cov_mat[0, 1]) < self.min_rho1:
            rho1_cov_mat = self.random_cov_mat()
        for i in range(self.num_stimuli):
            cov_mat[i] = rho1_cov_mat.copy()
        return cov_mat

    def _set_ps(self, means):
        return self._set_psa(self._set_psb(means))

    def _set_psa(self, means):
        # x-means of A are same across levels of B.
        offset = np.random.uniform(-self.min_mean_diff, self.min_mean_diff)
        means[4] = means[0] + offset
        offset = np.random.uniform(-self.min_mean_diff, self.min_mean_diff)
        means[6] = means[2] + offset
        return means


    def _set_psb(self, means):
        # y-means of B are same across levels of A
        offset = np.random.uniform(-self.min_mean_diff, self.min_mean_diff)
        means[3] = means[1] + offset
        offset = np.random.uniform(-self.min_mean_diff, self.min_mean_diff)
        means[5] = means[7] + offset
        return means

    def generate_parameter_controlled_cms(self, n_matrices, vary_means=True, vary_covariances=True, vary_crits=True):
        fixed_means = np.array([0., 0., 0.5, 0., 0., 0.5, 0.5, 0.5])
        fixed_cov_mat = np.array([np.eye(self.num_dimensions) for _ in range(self.num_stimuli)])
        fixed_crits = np.zeros(2, dtype=float)

        cms, trial_counts, params = [], [], []

        for i in tqdm(range(n_matrices), total=n_matrices, desc="Generating parameter controlled data"):
            n_trials = int(np.ceil(np.random.uniform(*self.trial_range)))
            means = self.random_means() if vary_means else fixed_means
            cov_mat = self._set_pi(self.get_cov_mats()) if vary_covariances else fixed_cov_mat
            c = self.random_crit(means) if vary_crits else fixed_crits

            samples = self.get_samples(means, cov_mat, n_trials, sample_loss_factor=self.sample_loss)
            cm_rows = []
            trial_counts_for_cm = np.array([])
            for _, stim_samples in enumerate(samples):
                response_counts = self.get_response_counts(stim_samples, c)
                cm_rows.append(response_counts)
                trial_counts_for_cm = np.concatenate([trial_counts_for_cm, [len(stim_samples)]])
            cm = np.vstack(cm_rows)
            flat_params = np.concatenate([means.flatten(), np.array([m.flatten() for m in cov_mat]).flatten(), c])
            cms.append(cm.flatten())
            trial_counts.append(trial_counts_for_cm)
            params.append(flat_params)
        return np.array(cms), np.array(params), np.array(trial_counts)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets for GRT modeling.")
    parser.add_argument("--full", action="store_true", help="Generate the full dataset with all model constraints.")
    parser.add_argument("--pretraining", action="store_true", help="Generate the parameter-controlled pre-training datasets.")
    parser.add_argument("--both", action="store_true", help="Generate both the full and pre-training datasets.")

    args = parser.parse_args()
    if not (args.full or args.pretraining or args.both):
        parser.error("At least one argument (--full, --pretraining, or --both) is required.")

    if args.both or args.pretraining:
        # Stage 1: Vary only means, keep everything else fixed
        print("Stage 1: Varying only means...")
        cms_stage1, params_stage1, trials_stage1 = GRTDataGenerator(num_matrices=NUM_PRETRAINING_MATRICES).generate_parameter_controlled_cms(
            n_matrices=NUM_PRETRAINING_MATRICES, 
            vary_means=True, 
            vary_covariances=False, 
            vary_crits=False
        )
        save_file_name = os.path.join(SIMULATED_DATA_DIR, "vary_m_dataset.npz")
        print(f"Generated {cms_stage1.shape[0]} and saved matrices for Stage 1 to {save_file_name}.")
        np.savez(save_file_name, X=cms_stage1, X_trials=trials_stage1, y_params=params_stage1)

        # Stage 2: Vary means and covariance matrices, keep crits fixed
        print("\nStage 2: Varying means and covariances...")
        cms_stage2, params_stage2, trials_stage2 = GRTDataGenerator(num_matrices=NUM_PRETRAINING_MATRICES).generate_parameter_controlled_cms(
            n_matrices=NUM_PRETRAINING_MATRICES, 
            vary_means=True, 
            vary_covariances=True, 
            vary_crits=False
        )
        save_file_name = os.path.join(SIMULATED_DATA_DIR, "vary_mv_dataset.npz")
        print(f"Generated {cms_stage2.shape[0]} and saved matrices for Stage 2 to {save_file_name}.")
        np.savez(save_file_name, X=cms_stage2, X_trials=trials_stage2, y_params=params_stage2)

        # Stage 3: Vary means and critical_values
        print("Stage 3: Varying means and critical values...")
        cms_stage3, params_stage3, trials_stage3 = GRTDataGenerator(num_matrices=NUM_PRETRAINING_MATRICES).generate_parameter_controlled_cms(
            n_matrices=NUM_PRETRAINING_MATRICES, 
            vary_means=True, 
            vary_covariances=False, 
            vary_crits=True
        )
        save_file_name = os.path.join(SIMULATED_DATA_DIR, "vary_mc_dataset.npz")
        print(f"Generated {cms_stage3.shape[0]} and saved matrices for Stage 3 to {save_file_name}.")
        np.savez(save_file_name, X=cms_stage3, X_trials=trials_stage3, y_params=params_stage3)

        # Stage 4: Vary the lot
        print("Stage 3: Varying the lot...")
        cms_stage4, params_stage4, trials_stage4 = GRTDataGenerator(num_matrices=NUM_PRETRAINING_MATRICES).generate_parameter_controlled_cms(
            n_matrices=NUM_PRETRAINING_MATRICES, 
            vary_means=True, 
            vary_covariances=True, 
            vary_crits=True
        )
        save_file_name = os.path.join(SIMULATED_DATA_DIR, "vary_mvc_dataset.npz")
        print(f"Generated {cms_stage4.shape[0]} and saved matrices for Stage 4 to {save_file_name}.")
        np.savez(save_file_name, X=cms_stage4, X_trials=trials_stage4, y_params=params_stage4)

    if args.both or args.full:
        # --- Existing data generation logic (unmodified for comparison) ---
        print("\n--- Generating data for all model constraints ---")
        gen = GRTDataGenerator(num_matrices=NUM_MATRICES_PER_MODEL, num_dimensions=2, num_levels=2, trial_range=TRIALS_RANGE)
        cms, parameters, trial_counts, y_cls, y_cls_label = gen.generate_all_model_cms()
        np.savez(DATASET_FILE, X=cms, X_trials=trial_counts, y_params=parameters, y_model_cls=y_cls, y_cls_label=y_cls_label)

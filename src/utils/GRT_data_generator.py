import numpy as np 
from scipy.stats import multivariate_normal
import os
from tqdm import tqdm
from src.utils.config import *
import argparse
from pprint import pprint as pp

class GRTDataGenerator:
    def __init__(self, num_matrices=10, num_dimensions=2, num_levels=2, trial_range=TRIALS_RANGE):
        self.num_matrices = num_matrices
        self.num_dimensions = num_dimensions
        self.num_levels = num_levels
        self.num_stimuli = self.num_dimensions * self.num_levels
        
        self.mean_range = (-5, 5) # setting an imbalance creates a greater chance of ordered means. Is this the best way of doing it? probably not even remotely.
        self.mean_sep = 0.2
        self.pi_tolerance = 0.2
        self.corr_range = (self.pi_tolerance, 0.99)
        self.variance = 1.0
        self.crit_range = (self.mean_range[0] * 0.9, self.mean_range[1] * 0.9)
        self.sample_loss = 0.1
        self.trial_range = trial_range
        self.min_accuracy = MIN_MATRIX_ACCURACY
        self.max_accuracy = MAX_MATRIX_ACCURACY
        self.n_accuracy_bins = MATRIX_ACCURACY_BINS
        
        self.model_names = MODEL_NAMES

    def accuracy_check(self, cm, min_acc, max_acc):
        correct_counts = np.diag(cm)
        total_trials = np.sum(cm, axis=1)
        accuracies = np.divide(
            correct_counts, total_trials, 
            out=np.zeros_like(correct_counts, dtype=float), 
            where=total_trials != 0
        )
        grand_mean_accuracy = np.mean(accuracies)
        return min_acc < grand_mean_accuracy <= max_acc

    def generate_cm(self, model_name, n_stimulus_trials, min_acc, max_acc):
        max_attempts = 10
        for n_trial_attempts in range(max_attempts):
            means, cov_mat = self.random_model_params(model_name, min_acc)
            cm = np.zeros((self.num_stimuli, self.num_stimuli), dtype=int)
            for sample_attempts in range(max_attempts):
                samples = self.get_samples(means, cov_mat, n_stimulus_trials, sample_loss_factor=self.sample_loss)
                for _ in range(max_attempts):
                    c = self.random_crit(means)
                    trial_counts = np.array([])
                    for i, stim_samples in enumerate(samples):
                        counts = self.get_response_counts(stim_samples, c)
                        cm[i, :] = counts
                        trial_counts = np.concatenate([trial_counts, [len(stim_samples)]])
                    if self.accuracy_check(cm, min_acc, max_acc):
                        flat_params = np.concatenate([means.flatten(), np.array([m.flatten() for m in cov_mat]).flatten(), c])
                        return cm, trial_counts, flat_params, means, cov_mat, c
            n_stimulus_trials+=1
        return None, None, None, None, None, None

    def generate_cms(self, model_name, n_matrices=1):
        cms, trial_counts, params = [], [], []
        min_accuracies = np.round(np.linspace(self.min_accuracy, self.max_accuracy, self.n_accuracy_bins) / 100, 3)
        for idx, min_acc in enumerate(min_accuracies[:-1]):
            max_acc = min_accuracies[idx+1]
            for i in tqdm(range(n_matrices), total=n_matrices, desc=f"Generating matrices for {model_name} at {min_acc*100:.1f}-{max_acc*100:.1f}%"):
                cm = None
                while cm is None:
                    n_trials = int(np.ceil(np.random.uniform(*self.trial_range)))
                    cm, trial_count, flat_params, _, _, _ = self.generate_cm(model_name, n_trials, min_acc, max_acc)
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
            y_cls.extend([model_class]*len(model_cms))
            y_cls_label.extend([model_label]*len(model_cms))
         
        return np.array(cms), np.array(parameters), np.array(trial_counts), np.array(y_cls), np.array(y_cls_label)
    
    def random_model_params(self, model_name="pi_ps_ds", min_acc=0.25):
        has_pi = 'pi' in model_name
        has_rho1 = 'rho1' in model_name
        has_psa = 'psa' in model_name or 'ps_' in model_name
        has_psb = 'psb' in model_name or 'ps_' in model_name
        means = self._generate_constrained_means(has_psa, has_psb, min_acc)
        cov_mat = self._generate_constrained_cov(has_pi, has_rho1)
        return means, cov_mat#, c


    def _generate_constrained_means(self, has_psa, has_psb, min_acc):
        while True:
            means_x = np.hstack([0.0, np.random.uniform(*self.mean_range, self.num_stimuli-1)])
            means_y = np.hstack([0.0, np.random.uniform(*self.mean_range, self.num_stimuli-1)])
            if np.random.rand() < min_acc:
                # order x means
                means_x[1] = np.random.uniform(0, self.mean_range[1])
                means_x[3] = np.random.uniform(means_x[2], self.mean_range[1])
                # order y means
                means_y[2] = np.random.uniform(0, self.mean_range[1])
                means_y[3] = np.random.uniform(means_y[1], self.mean_range[1])
                
            # Apply PSA constraint
            if has_psa:
                means_x[2] = means_x[0]
                means_x[1] = means_x[3]
            # Apply PSB constraint
            if has_psb:
                means_y[1] = means_y[0]
                means_y[2] = means_y[3]
            
            # Check for accidental perfect constraints
            psa_accident = not has_psa and np.abs(means_x[2] - means_x[0]) < self.mean_sep and np.abs(means_x[3] - means_x[1]) < self.mean_sep
            psb_accident = not has_psb and np.abs(means_y[1] - means_y[0]) < self.mean_sep and np.abs(means_y[3] - means_y[2]) < self.mean_sep
            
            if not psa_accident and not psb_accident:
                means = np.vstack([means_x, means_y]).T.flatten()
                return np.round(means, 3)

    

    def _create_cov_mat(self, corr):
        return np.array([
            [self.variance, corr*self.variance],
            [corr*self.variance, self.variance]
        ])

    def _generate_constrained_cov(self, has_pi, has_rho1):
        if has_pi:
            return self._set_pi()
        elif has_rho1:
            return self._set_rho1()
        else: 
            return self._set_ds()

    def _set_pi(self):
        pi_matrix = self._create_cov_mat(0.0)
        return np.array([pi_matrix] * self.num_stimuli)

    def _set_rho1(self):
        corr = np.random.uniform(*self.corr_range)
        if np.random.rand() < 0.5:
            corr = -corr
        rho1_matrix = self._create_cov_mat(corr)
        return np.array([rho1_matrix] * self.num_stimuli) 

    def _set_ds(self):
        correlations = []
        while True:
            corrs = np.random.uniform(*self.corr_range, self.num_stimuli)
            signs = np.random.choice([-1, 1], self.num_stimuli)
            correlations = corrs * signs
            
            # Check for pi-like accident (all near zero) and rho1-like accident (all similar)
            is_pi_like = np.all(np.abs(correlations) < self.pi_tolerance)
            is_rho1_like = np.max(correlations) - np.min(correlations) < self.pi_tolerance
            if not is_pi_like and not is_rho1_like:
                break
                
        cov_mats = [self._create_cov_mat(c) for c in correlations]
        return np.array(cov_mats)

    def random_crit(self, means):
        """
        Generates critical values from a normal distribution centered on the mean
        of the stimulus locations for each dimension.
        """
        # Get the mean positions for each dimension
        means_x = means[0::2]
        means_y = means[1::2]
        
        # Calculate the center and standard deviation for the normal distribution
        # The standard deviation is a fraction of the range of the means.
        mid_x = np.mean(means_x)
        mid_y = np.mean(means_y)
        std_x = (np.max(means_x) - np.min(means_x)) / (len(means_x) + 1)
        std_y = (np.max(means_y) - np.min(means_y)) / (len(means_y) + 1)

        c_x = np.random.normal(mid_x, std_x)
        c_y = np.random.normal(mid_y, std_y)
        
        c_x = np.clip(c_x, self.crit_range[0], self.crit_range[1])
        c_y = np.clip(c_y, self.crit_range[0], self.crit_range[1])
        
        return np.round(np.array([c_x, c_y]), 2)
    
    
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
        counts[1] = np.sum((samples[:, 0] >= c[0]) & (samples[:, 1] < c[1]))
        counts[2] = np.sum((samples[:, 0] < c[0]) & (samples[:, 1] >= c[1]))
        counts[3] = np.sum((samples[:, 0] >= c[0]) & (samples[:, 1] >= c[1]))
        return counts
   
    
    def generate_parameter_controlled_cms(self, n_matrices, vary_means=True, vary_covariances=True, vary_crits=True):
        fixed_means = np.array([0., 0., 0.5, 0., 0., 0.5, 0.5, 0.5])
        fixed_cov_mat = np.array([np.eye(self.num_dimensions) for _ in range(self.num_stimuli)])
        fixed_crits = np.zeros(2, dtype=float)

        cms, trial_counts, params = [], [], []

        for i in tqdm(range(n_matrices), total=n_matrices, desc="Generating parameter controlled data"):
            n_trials = int(np.ceil(np.random.uniform(*self.trial_range)))
            means = self.random_means() if vary_means else fixed_means
            cov_mat = self._set_pi() if vary_covariances else fixed_cov_mat
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

    def generate_trial_data(self, model_name, n_trials=1000):
        """
        Intended to be used for training an LSTM.
        Generates trial-by-trial data for a single subject, given a ground-truth model.
        Args:
            model_name (str): The name of the GRT model to use for data generation.
            n_trials (int): The total number of trials to simulate.
        Returns:
            tuple: A tuple containing:
                - trials (np.array): An array of shape (n_trials, 2) where each row is
                  [stimulus_index, response_index].
                - flat_params (np.array): A flattened array of the ground-truth parameters.
                - model_name (str): The model name for this dataset.
        """
        # Get the ground truth parameters for the specified model
        means, cov_mat, c = self.random_model_params(model_name)
        flat_params = np.concatenate([means.flatten(), np.array([m.flatten() for m in cov_mat]).flatten(), c])

        # Generate a list of stimuli to be presented, in random order
        # ensuring a balanced presentation of stimuli across trials
        stimulus_sequence = np.random.choice(self.num_stimuli, size=n_trials, replace=True)

        trials = []
        for trial_num in tqdm(range(n_trials), desc=f"Generating trials for {model_name}"):
            stimulus_idx = stimulus_sequence[trial_num]
            sample = multivariate_normal.rvs(
                mean=means[stimulus_idx*2 : stimulus_idx*2+2],
                cov=cov_mat[stimulus_idx],
                size=1
            )
            if sample[0] < c[0]:
                if sample[1] < c[1]:
                    response_idx = 0
                else:
                    response_idx = 1
            else:
                if sample[1] < c[1]:
                    response_idx = 2
                else:
                    response_idx = 3
            trials.append([stimulus_idx, response_idx])
        return np.array(trials), flat_params, model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets for GRT modeling.")
    parser.add_argument("--full", action="store_true", help="Generate the full dataset with all model constraints.")
    parser.add_argument("--pretraining", action="store_true", help="Generate the parameter-controlled pre-training datasets.")
    parser.add_argument("--tbt", action="store_true", help="Generate trial-by-trial data for LSTM training (or for whatever purpose you might want trial-by-trial data).")
    parser.add_argument("--all", action="store_true", help="We'll take the lot! Generate the full, pre-training, and trial-by-trial datasets.")

    args = parser.parse_args()
    if not (args.full or args.pretraining or args.all or args.tbt):
        parser.error("At least one argument (--full, --pretraining, or --tbt, or --all) is required.")

    if args.all or args.pretraining:
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

    if args.all or args.full:
        # --- Existing data generation logic (unmodified for comparison) ---
        print("\n--- Generating data for all model constraints ---")
        gen = GRTDataGenerator(num_matrices=NUM_MATRICES_PER_ACCURACY_BIN, num_dimensions=2, num_levels=2, trial_range=TRIALS_RANGE)
        cms, parameters, trial_counts, y_cls, y_cls_label = gen.generate_all_model_cms()
        np.savez(DATASET_FILE, X=cms, X_trials=trial_counts, y_params=parameters, y_model_cls=y_cls, y_cls_label=y_cls_label)

    if args.all or args.tbt:
        print("\n--- Generating trial-by-trial data ---")
        gen = GRTDataGenerator(num_matrices=NUM_MATRICES_PER_MODEL, num_dimensions=2, num_levels=2, trial_range=TRIALS_RANGE)
        
        all_trials = []
        all_params = []
        all_labels = []
        
        num_sequences_per_model = 500
        
        for model_label in gen.model_names:
            print(f"Generating {num_sequences_per_model} trial sequences for model: {model_label}")
            for _ in tqdm(range(num_sequences_per_model)):
                trials, params, label = gen.generate_trial_data(model_label, n_trials=1000)
                all_trials.append(trials)
                all_params.append(params)
                all_labels.append(label)
                
        np.savez(
            TRIAL_BY_TRIAL_FIAL,
            X=all_trials,
            y_params=all_params,
            y_model_labels=all_labels
        )
        print(f"Done! Trial-by-trial data saved to {TRIAL_BY_TRIAL_FIAL}.")


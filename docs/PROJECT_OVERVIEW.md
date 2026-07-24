GRIN: General Recognition Inference Network
A Framework for Simulation-Based Inference and Adaptive Perceptual Training

Version: 2.0 (Post-Architectural Review)
Status: Active Development
Last Updated: July 2026
Executive Summary

GRIN (General Recognition Inference Network) is a framework for simulation-based inference (SBI) applied to General Recognition Theory (GRT) models. The system uses neural networks trained on simulated data to perform fast, amortized inference of perceptual representations.

Unlike traditional GRT model fitting, which requires seconds to minutes per dataset via maximum likelihood estimation (MLE), GRIN provides parameter estimates in milliseconds. This speed enables real-time tracking of perceptual representations and opens the door to adaptive perceptual training—a closed-loop system where inferred representations guide stimulus selection to optimize learning.

Key Insight: SBI is the natural computational framework for adaptive training because it provides:

    Speed - Real-time parameter updates

    Uncertainty - Quantification of what we don't know

    Amortization - Train once, use for any participant

    Flexibility - Handles arbitrary model structures

Background and Motivation
The Problem: From Passive Analysis to Active Intervention

General Recognition Theory (GRT; Ashby & Townsend, 1986) provides a rich framework for understanding perceptual representations. GRT models characterize how observers perceive multidimensional stimuli, capturing:

    Perceptual means - Where stimuli are perceived in perceptual space

    Covariance structure - Feature independence vs. correlation

    Decision boundaries - How observers categorize stimuli

The Current Bottleneck: Fitting GRT models to behavioral data is computationally expensive. Maximum likelihood estimation (MLE) or Bayesian methods can take seconds to minutes per model, making real-time inference impossible. This limits GRT to post-hoc analysis rather than active intervention.

The Opportunity: If we can perform real-time inference of perceptual representations, we can:

    Track how representations change during learning

    Identify specific perceptual confusions

    Select stimuli that target those confusions

    Optimize training for each individual

The Solution: Simulation-Based Inference

Simulation-based inference (SBI; Cranmer et al., 2020) is a class of methods where neural networks learn to invert simulators. Instead of performing expensive likelihood optimization for each new dataset, an SBI network learns the mapping from data to parameters through supervised learning on simulated data.

Application to GRT:

    Simulate - Generate large-scale GRT datasets with known parameters

    Learn - Train neural networks to predict parameters from data

    Infer - Use the trained network for fast parameter estimation

This is GRIN: a neural network that maps confusion matrices to GRT parameters in milliseconds.
The Vision: Adaptive Perceptual Training

GRIN's speed enables a transformative application: adaptive perceptual training that uses real-time inference to guide stimulus selection.

The closed-loop system operates as:
text

Participant → Responses → GRIN Inference → Perceptual Representation (θ̂)
                                              ↓
                                         Stimulus Selection
                                              ↓
                                       Next Stimulus → Participant

Why This Works: By knowing the participant's current perceptual representation, we can:

    Identify specific confusions (e.g., correlations between features)

    Select stimuli that isolate those confusions

    Maximize information gain or learning progress

    Track improvement in real-time

This reframes GRT from a descriptive tool to an intervention engine.
Foundational Decisions and Rationale
Decision 1: Regression-First, Classification-Second

The Question: Should we predict the model class first, then parameters, or parameters first, then model class?

Our Decision: Regression-first (predict parameters, then infer model class from parameters).

Rationale:
Approach	Pros	Cons	For Adaptive Training?
Classify First	Clean separation, interpretable	Error propagation, binary decisions	❌ Too brittle
Regress First	Continuous updates, flexible, uncertainty	Harder to train	✅ Perfect
Joint Multi-Task	Single model, shared features	Task conflict, hard to balance	⚠️ Possible but fragile

Why Regress-First Wins:

    Solves the PI/RHO identifiability problem - Predicting correlations first reveals whether they're zero or equal

    Provides uncertainty estimates - Essential for adaptive stimulus selection

    Enables continuous updates - Parameters update smoothly with new data

    More scientific - GRT parameters are the fundamental quantities

Implementation:
python

# Step 1: Predict parameters (the hard part)
params = grin_model.predict(confusion_matrix)  # 26 parameters

# Step 2: Infer model class from parameters
model_class = infer_class_from_params(params)
# Check: Are correlations ~0? → PI. Are all correlations similar? → RHO1. etc.

Decision 2: Simulation-Based Inference as the Core Methodology

The Question: Should we use MLE (traditional) or SBI (neural network) for inference?

Our Decision: SBI (simulation-based inference).

Rationale:
Aspect	Traditional MLE	SBI (GRIN)
Speed per fit	2-30 seconds	5-50 milliseconds
Scalability	Fit each participant separately	Train once, use everywhere
Real-time updates	Impossible	Possible
Adaptive training	Too slow	Fast enough
Uncertainty	Asymptotic approximations	MC Dropout, ensembles

Why SBI Wins for Adaptive Training:

    Speed - Fast enough for block-by-block updates

    Amortization - Train on simulations once, use for all participants

    Flexibility - Can handle any GRT model structure

    Uncertainty - Natural uncertainty estimates via MC Dropout

The Synergy: SBI is not just a convenient method—it's the computational framework that makes adaptive training possible.

Decision 3: Uncertainty Quantification Via MC Dropout

The Question: How do we quantify uncertainty in parameter estimates?

Our Decision: MC Dropout (Gal & Ghahramani, 2016).

Rationale:
Method	Pros	Cons
Bayesian Neural Networks	Full posterior	Complex, slow, hard to train
Ensembles	Good uncertainty	Expensive (multiple models)
MC Dropout	Simple, fast, works	Approximate
Laplace Approximation	Fast	Asymptotic, not always accurate

Why MC Dropout Wins:

    Simple to implement - Just keep dropout active at inference

    Fast - Single model, multiple forward passes

    Works with existing code - Minimal changes required

    Good enough - Provides reasonable uncertainty estimates

Implementation:
python

def predict_with_uncertainty(model, X, n_samples=50):
    predictions = []
    for _ in range(n_samples):
        pred = model(X, training=True)  # Dropout active
        predictions.append(pred)
    return np.mean(predictions), np.std(predictions)

Decision 4: Stimulus Selection via Information Gain

The Question: How do we select the next stimulus to maximize learning?

Our Decision: Information Gain (maximize expected reduction in uncertainty).

Rationale:
Strategy	Pros	Cons
Random	Simple	Inefficient
Uncertainty Sampling	Targets uncertain regions	May ignore structure
Information Gain	Theoretically optimal	Computationally expensive
Active Learning	General framework	Complex

Why Information Gain Wins:

    Theoretically grounded - Maximizes learning

    Adapts to participant - Personalizes training

    Balances exploration/exploitation - Natural trade-off

    Uses uncertainty estimates - Leverages MC Dropout

Implementation:
python

def compute_information_gain(model, stimulus, current_params):
    # Simulate responses to this stimulus
    responses = simulate_responses(stimulus, current_params, n=100)
    
    # Get parameter predictions for each response
    predicted_params = [model.predict(r) for r in responses]
    
    # Compute entropy reduction
    current_entropy = entropy(current_params)
    expected_entropy = mean([entropy(p) for p in predicted_params])
    
    return current_entropy - expected_entropy  # Information gain

Decision 5: Hierarchical vs. Unified Architecture

The Question: Should we use a single general regressor or specialist regressors per model?

Our Decision: Unified general regressor with uncertainty.

Rationale:
Architecture	Pros	Cons
Specialist Regressors	More accurate per model	Need 12 models, error propagation
General Regressor	One model, flexible	Harder to train
Hierarchical (Class+Reg)	Best of both	Complex, cascading errors

Why Unified Wins:

    No error propagation - No classification step to get wrong

    Continuous parameter space - Predicts any parameter value

    Simpler to maintain - One model, one training pipeline

    Supports uncertainty - Uncertainty captures model ambiguity

The PI/RHO Problem Solved:
text

Traditional Approach: Classify first
Data → Classifier → "PI" → PI Specialist → Correlation = 0
                                    (If classifier wrong, everything fails)

Our Approach: Regress first
Data → Regressor → Correlation = 0.05 → Infer: "Probably PI, but uncertain"
                                    (Even if correlation is ambiguous, we recover it)

The GRIN Architecture
Overview

GRIN is a neural network trained on simulated GRT data to perform fast inference of perceptual parameters.
text

Input: Confusion Matrix (16 numbers)
↓
[Feature Extraction] → Log proportions, trial counts, entropy, etc.
↓
[Neural Network] → Multi-layer perceptron with residual connections
↓
Output: 26 GRT Parameters
    - 8 Means (4 stimuli × 2 dimensions)
    - 16 Covariance elements (4 stimuli × 4 elements)
    - 2 Criteria (decision bounds)

Key Components
1. Data Generation

The GRT data generator simulates confusion matrices from known parameters:
python

class GRTDataGenerator:
    def generate_cm(self, model_name, n_trials, target_accuracy):
        # Generate parameters from model constraints
        means, cov_mat = self.random_model_params(model_name)
        # Calibrate to target accuracy
        scaled_means, cm, c = self.calibrate_means_and_crit(...)
        return cm, scaled_means, cov_mat, c

Key Features:

    12 GRT model types (PI, RHO1, DS with various PS combinations)

    4 stimulus types (2×2 factorial design)

    Accuracy calibration (25-100% range)

    Trial count variation (1-1000 trials per stimulus)

2. Neural Network Architecture

The base model is a multi-layer perceptron with:
python

Input (20 features) → Dense(512) → Residual Block(512) → Dense(256) 
    → Dense(128) → Output Heads:
        - Means: Dense(8, linear)
        - Covariances: Dense(16, linear)
        - Criteria: Dense(2, linear)

Architectural Variants:

    Shared backbone (all tasks share features)

    Parallel backbones (separate features for each task)

    MC Dropout (for uncertainty)

Why This Architecture:

    Residual blocks enable deeper networks

    Dropout prevents overfitting

    Separate output heads allow task-specific scaling

3. Training Strategy

Curriculum Learning: Progressive exposure to model complexity:
python

STAGES = [
    Stage 1: Simple models (PI, RHO1 with PS constraints)
    Stage 2: Intermediate models (PSA, PSB)
    Stage 3: Full complexity models (DS)
    Stage 4: All models mixed
]

Why Curriculum Learning:

    Easier to learn basic patterns first

    Prevents catastrophic forgetting

    Builds robust representations

4. Inference and Uncertainty
python

def infer_representation(responses, model):
    # Convert responses to confusion matrix
    cm = build_confusion_matrix(responses)
    
    # Get parameter predictions with uncertainty
    params_mean, params_std = model.predict_with_uncertainty(cm, n_samples=50)
    
    return {
        'means': params_mean[:8],
        'covariances': params_mean[8:24],
        'criteria': params_mean[24:26],
        'uncertainty': params_std
    }

Model Class Inference

Rather than training a separate classifier, we infer model class from predicted parameters:
python

def infer_model_class(params):
    """
    Infer GRT model class from parameter predictions.
    """
    correlations = extract_correlations(params['covariances'])
    
    # Check for PI (all correlations ~0)
    if np.all(np.abs(correlations) < 0.1):
        covariance_type = 'PI'
    # Check for RHO1 (all correlations similar)
    elif np.std(correlations) < 0.05:
        covariance_type = 'RHO1'
    # Otherwise, DS (correlations differ)
    else:
        covariance_type = 'DS'
    
    # Infer mean constraints (PS, PSA, PSB)
    means = params['means'].reshape(4, 2)
    if np.allclose(means[1,0], means[0,0]) and np.allclose(means[3,0], means[2,0]):
        mean_type = 'PSA'
    elif np.allclose(means[2,1], means[0,1]) and np.allclose(means[3,1], means[1,1]):
        mean_type = 'PSB'
    else:
        mean_type = 'DS'
    
    return f"{covariance_type}_{mean_type}_DS"

Why This Works:

    Uses the full parameter information

    Natural handling of ambiguous cases

    Provides confidence from uncertainty estimates

The Adaptive Training System
Overview

The adaptive training system uses GRIN's fast inference to select stimuli that maximize learning.
text

┌─────────────────────────────────────────────────────────────┐
│                    Adaptive Training Loop                    │
├─────────────────────────────────────────────────────────────┤
│  1. Present stimulus to participant                         │
│  2. Collect response                                        │
│  3. Update confusion matrix                                 │
│  4. Infer parameters via GRIN (fast)                       │
│  5. Compute uncertainty                                     │
│  6. Select next stimulus maximizing information gain       │
│  7. Repeat                                                 │
└─────────────────────────────────────────────────────────────┘

Stimulus Selection Strategies
Strategy 1: Uncertainty Reduction

Select stimuli that maximize reduction in parameter uncertainty:
python

def select_by_uncertainty_reduction(model, current_params, stimulus_pool):
    best_stimulus = None
    best_reduction = -np.inf
    
    for stimulus in stimulus_pool:
        # Predict response to this stimulus
        predicted_response = model.simulate_response(stimulus, current_params)
        
        # Update parameters with this response
        new_params = model.predict(predicted_response)
        
        # Compute uncertainty reduction
        current_uncertainty = np.mean(current_params['uncertainty'])
        new_uncertainty = np.mean(new_params['uncertainty'])
        reduction = current_uncertainty - new_uncertainty
        
        if reduction > best_reduction:
            best_reduction = reduction
            best_stimulus = stimulus
    
    return best_stimulus

Strategy 2: Target Representation Alignment

Select stimuli that push representation toward a target (expert) representation:
python

def select_by_target_alignment(model, current_params, target_params, stimulus_pool):
    best_stimulus = None
    best_improvement = -np.inf
    
    for stimulus in stimulus_pool:
        # Predict response to this stimulus
        predicted_response = model.simulate_response(stimulus, current_params)
        
        # Update parameters with this response
        new_params = model.predict(predicted_response)
        
        # Compute distance to target
        current_distance = distance(current_params, target_params)
        new_distance = distance(new_params, target_params)
        improvement = current_distance - new_distance
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_stimulus = stimulus
    
    return best_stimulus

Strategy 3: Prediction Error / Surprise

Select stimuli that maximize prediction error (learning signal):
python

def select_by_prediction_error(model, current_params, stimulus_pool):
    best_stimulus = None
    best_error = -np.inf
    
    for stimulus in stimulus_pool:
        # Predict response to this stimulus
        predicted_response = model.simulate_response(stimulus, current_params)
        
        # Simulate observing an actual response
        # (in practice, this would come from the participant)
        
        # Compute prediction error
        actual_response = participant.respond(stimulus)  # Real response
        error = cross_entropy(predicted_response, actual_response)
        
        if error > best_error:
            best_error = error
            best_stimulus = stimulus
    
    return best_stimulus

Unified Utility Function

Combine strategies with weights:
python

def compute_utility(stimulus, current_params, model, target_params=None):
    """
    Compute combined utility for stimulus selection.
    """
    # Information gain component
    info_gain = compute_information_gain(model, stimulus, current_params)
    
    # Target alignment component
    if target_params is not None:
        alignment = compute_target_alignment(model, stimulus, current_params, target_params)
    else:
        alignment = 0
    
    # Prediction error component
    prediction_error = compute_prediction_error(model, stimulus, current_params)
    
    # Weighted combination
    utility = (
        λ_info * info_gain +
        λ_alignment * alignment +
        λ_error * prediction_error
    )
    
    return utility

Validation: Simulated Learners

Before human experiments, validate the system on simulated learners:
python

class SimulatedLearner:
    """
    Simulates a participant with known GRT parameters.
    """
    def __init__(self, true_params, learning_rate=0.01):
        self.true_params = true_params
        self.current_params = true_params.copy()  # Start at true
        self.learning_rate = learning_rate
    
    def respond(self, stimulus):
        # Generate response from current parameters
        response = simulate_response(stimulus, self.current_params)
        # Update parameters based on learning
        self.current_params += self.learning_rate * self.get_feedback(stimulus, response)
        return response

def evaluate_adaptive_training():
    # Initialize simulated learner
    learner = SimulatedLearner(true_params=ground_truth)
    
    # Run adaptive training
    history = []
    for trial in range(n_trials):
        # Select stimulus
        stimulus = adaptive_selection(learner.current_params)
        
        # Get response
        response = learner.respond(stimulus)
        
        # Track progress
        history.append({
            'trial': trial,
            'params': learner.current_params.copy(),
            'stimulus': stimulus,
            'response': response
        })
    
    # Evaluate learning
    final_error = distance(learner.current_params, ground_truth)
    return history, final_error

Evaluation Framework
Parameter Recovery

Measure how well GRIN recovers true parameters:
python

def evaluate_parameter_recovery(grin_model, test_data):
    """
    Evaluate parameter recovery accuracy.
    """
    results = []
    for cm, true_params in test_data:
        pred_params, uncertainty = grin_model.predict_with_uncertainty(cm)
        
        results.append({
            'mae': mean_absolute_error(true_params, pred_params),
            'correlation': pearson_correlation(true_params, pred_params),
            'uncertainty': uncertainty.mean(),
            'coverage': compute_coverage_interval(true_params, pred_params, uncertainty)
        })
    
    return pd.DataFrame(results)

Model Identification

Measure how well inferred model class matches ground truth:
python

def evaluate_model_identification(grin_model, test_data):
    """
    Evaluate model class identification accuracy.
    """
    true_classes = []
    inferred_classes = []
    uncertainties = []
    
    for cm, true_class, true_params in test_data:
        pred_params, uncertainty = grin_model.predict_with_uncertainty(cm)
        inferred_class = infer_model_class(pred_params)
        
        true_classes.append(true_class)
        inferred_classes.append(inferred_class)
        uncertainties.append(uncertainty.mean())
    
    return {
        'accuracy': accuracy_score(true_classes, inferred_classes),
        'confusion_matrix': confusion_matrix(true_classes, inferred_classes),
        'uncertainty_by_class': group_uncertainty(true_classes, uncertainties)
    }

Adaptive Training Performance

Evaluate learning improvement under adaptive vs. random selection:
python

def compare_training_strategies():
    """
    Compare adaptive vs random stimulus selection.
    """
    strategies = {
        'adaptive': AdaptiveSelector(grin_model),
        'random': RandomSelector(),
        'uncertainty': UncertaintySelector(grin_model),
        'target_alignment': TargetSelector(grin_model, target_params)
    }
    
    results = {}
    for name, strategy in strategies.items():
        learner = SimulatedLearner(ground_truth)
        history = run_training(learner, strategy, n_trials=500)
        results[name] = {
            'final_error': history[-1]['error'],
            'learning_curve': [h['error'] for h in history],
            'stimulus_usage': analyze_stimulus_usage(history)
        }
    
    return results

Benchmarking Against Traditional Methods

Compare GRIN to traditional MLE fitting:
python

def benchmark_against_mle(test_data):
    """
    Compare GRIN (SBI) vs MLE (grtools).
    """
    results = {
        'sbi': {'times': [], 'errors': [], 'models': []},
        'mle': {'times': [], 'errors': [], 'models': []}
    }
    
    for cm, true_params, true_class in test_data:
        # SBI (GRIN)
        start = time.time()
        pred_params, uncertainty = grin_model.predict_with_uncertainty(cm)
        sbi_time = time.time() - start
        sbi_error = mean_absolute_error(true_params, pred_params)
        
        # MLE (grtools)
        start = time.time()
        mle_fit = grt_hm_fit(cm)
        mle_time = time.time() - start
        mle_error = mean_absolute_error(true_params, mle_fit.parameters)
        
        results['sbi']['times'].append(sbi_time)
        results['sbi']['errors'].append(sbi_error)
        results['mle']['times'].append(mle_time)
        results['mle']['errors'].append(mle_error)
    
    return results

Expected Results:

    Speed: GRIN (5-50ms) vs MLE (2-30s) → 100-1000x faster

    Accuracy: Comparable for clear cases, GRIN better for ambiguous due to uncertainty

    Scalability: GRIN handles large datasets, MLE struggles

Research Questions and Future Directions
Current Research Questions

    Identifiability: How well can we distinguish PI from RHO1 models?

        Hypothesis: Regression-first with uncertainty will outperform classification-first

        Test: Compare recovery accuracy across model types

    Uncertainty Calibration: Are MC Dropout uncertainty estimates well-calibrated?

        Hypothesis: Uncertainty will correlate with prediction error

        Test: Compute calibration plots (error vs. uncertainty)

    Adaptive Training Efficiency: Does adaptive selection outperform random selection?

        Hypothesis: Adaptive training will reduce learning time by ≥30%

        Test: Simulated learner experiments

    Transferability: Does GRIN generalize across parameter ranges?

        Hypothesis: Training on broad ranges enables generalization

        Test: Test on held-out parameter ranges

Future Directions

    Trial-Level Modeling

        Extend to trial-by-trial data (LSTM, RNN architectures)

        Enable even finer-grained adaptation

    Bayesian Neural Networks

        Replace MC Dropout with variational inference

        Provide proper Bayesian uncertainty

    Stimulus Optimization

        Use gradient-based optimization for stimulus design

        Generate novel stimuli rather than selecting from a pool

    Human Validation

        Run human experiments to validate adaptive training

        Compare to traditional training methods

    Multi-Dimensional Stimuli

        Extend beyond 2×2 factorial designs

        Handle arbitrary numbers of dimensions and levels

    Neuroimaging Integration

        Combine behavioral inference with neuroimaging data

        Validate representations with brain activity

Practical Implementation Guide
Installation
bash

# Clone repository
git clone https://github.com/murraysbennett/grin.git
cd grin

# Install dependencies
pip install -r requirements.txt

Basic Usage: Inference
python

from grin import GRINModel
import numpy as np

# Load pre-trained model
model = GRINModel.load('pretrained_models/grin_v2.h5')

# Load confusion matrix
cm = np.load('data/participant_cm.npy')  # Shape: (4, 4)

# Infer parameters
params, uncertainty = model.predict_with_uncertainty(cm, n_samples=50)

# Extract components
means = params[:8].reshape(4, 2)
covariances = params[8:24].reshape(4, 2, 2)
criteria = params[24:26]

# Infer model class
model_class = infer_model_class(means, covariances)

print(f"Model Class: {model_class}")
print(f"Parameter Uncertainty: {uncertainty.mean():.3f}")

Basic Usage: Training
python

from grin import GRINDataGenerator, GRINTrainer

# Generate data
generator = GRTDataGenerator()
data = generator.generate_all_model_cms(n_matrices=10000)

# Train model
trainer = GRINTrainer()
model = trainer.train(
    X=data['confusion_matrices'],
    y=data['parameters'],
    epochs=200,
    batch_size=128
)

# Save model
model.save('models/grin_trained.h5')

Advanced Usage: Adaptive Training
python

from grin import AdaptiveTrainer, StimulusPool

# Initialize
trainer = AdaptiveTrainer(
    model=grin_model,
    stimulus_pool=StimulusPool.from_dataset('stimuli.npy'),
    update_interval=20  # Update every 20 trials
)

# Run session
responses = []
for trial in range(500):
    # Select stimulus
    stimulus = trainer.select_stimulus()
    
    # Present stimulus and collect response (from participant)
    response = participant.respond(stimulus)
    responses.append(response)
    
    # Update model
    if trial % trainer.update_interval == 0:
        trainer.update_model(responses)

# Final inference
final_params = trainer.infer_final(responses)

Theoretical Contributions
1. SBI for GRT

This project establishes simulation-based inference as a viable alternative to MLE for GRT model fitting. Key contributions:

    Demonstration of SBI speed and accuracy

    Handling of complex model structures

    Integration of uncertainty quantification

2. Adaptive Training Framework

This project provides a blueprint for adaptive perceptual training:

    Real-time representation tracking

    Information-theoretic stimulus selection

    Closed-loop learning optimization

3. Bridging Cognitive Modeling and AI

This project demonstrates how cognitive models (GRT) can be combined with AI methods (SBI) to create new applications:

    From passive analysis to active intervention

    From group-level to individual-level adaptation

    From description to prescription

References

Ashby, F. G., & Townsend, J. T. (1986). Varieties of perceptual independence. Psychological Review, 93(2), 154-179.

Cranmer, K., Brehmer, J., & Louppe, G. (2020). The frontier of simulation-based inference. Proceedings of the National Academy of Sciences, 117(48), 30055-30062.

Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. International Conference on Machine Learning, 1050-1059.

Silbert, N. H., & Thomas, R. D. (2013). Decisional separability, model identification, and statistical inference in the general recognition theory framework. Psychonomic Bulletin & Review, 20(1), 1-20.

Thomas, R. D. (2001). Perceptual interactions of facial dimensions in speeded classification and identification. Perception & Psychophysics, 63(4), 625-650.
Appendices
Appendix A: GRT Model Definitions
Model	Covariance Type	Mean Type	Criteria Type
PI_PS_DS	PI (all 0)	PS (full)	DS
PI_PSA_DS	PI (all 0)	PSA (partial)	DS
PI_PSB_DS	PI (all 0)	PSB (partial)	DS
RHO1_PS_DS	RHO1 (equal)	PS (full)	DS
RHO1_PSA_DS	RHO1 (equal)	PSA (partial)	DS
RHO1_PSB_DS	RHO1 (equal)	PSB (partial)	DS
PI_DS	PI (all 0)	DS (none)	DS
PS_DS	DS (diff)	PS (full)	DS
RHO1_DS	RHO1 (equal)	DS (none)	DS
PSA_DS	DS (diff)	PSA (partial)	DS
PSB_DS	DS (diff)	PSB (partial)	DS
DS	DS (diff)	DS (none)	DS
Appendix B: Parameter Structure
text

Parameters (26 total):
  Means (8): 
    μ₁ₓ, μ₁ᵧ, μ₂ₓ, μ₂ᵧ, μ₃ₓ, μ₃ᵧ, μ₄ₓ, μ₄ᵧ
  
  Covariance Matrices (16):
    Σ₁: [σ₁ₓ², σ₁ₓᵧ; σ₁ₓᵧ, σ₁ᵧ²]
    Σ₂: [σ₂ₓ², σ₂ₓᵧ; σ₂ₓᵧ, σ₂ᵧ²]
    Σ₃: [σ₃ₓ², σ₃ₓᵧ; σ₃ₓᵧ, σ₃ᵧ²]
    Σ₄: [σ₄ₓ², σ₄ₓᵧ; σ₄ₓᵧ, σ₄ᵧ²]
  
  Criteria (2):
    cₓ, cᵧ

Appendix C: Software Dependencies
    Python: ≥3.9
    TensorFlow: ≥2.10
    NumPy: ≥1.22
    Scikit-learn: ≥1.0
    Matplotlib: ≥3.5
    Seaborn: ≥0.11
    Pandas: ≥1.4
    R (optional): grtools, tidyverse, here

Acknowledgments

This project builds on foundational work in General Recognition Theory by Ashby, Townsend, Thomas, and Silbert. The simulation-based inference framework draws from developments in machine learning and computational statistics. The adaptive training vision is inspired by applications in medical imaging, perceptual expertise, and educational psychology.

Contact: Murray S. Bennett, bennett.1755@osu.edu
Project Repository: https://github.com/murraysbennett/grin

This document serves as the guiding framework for GRIN development. It will be updated as the project evolves and as new findings emerge.
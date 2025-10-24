import os
import sys
import math
import numpy as np
from tqdm import tqdm
import torch
from eval import read_run_dir, get_model_from_run
from tasks import FourierSeriesV3
import copy
from joblib import Parallel, delayed

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, default='../main')
    parser.add_argument('--output-dir', type=str, default='prepare_benchmark')

    return parser.parse_args()

L = 5


def ridge_single_problem(X_i, y_i, lambd):
    """
    Solve ridge regression for a single problem.

    Parameters:
    X_i (numpy.ndarray): Input features of shape (n_samples, n_features).
    y_i (numpy.ndarray): Target values of shape (n_samples,).
    lambd (float): Regularization parameter (lambda).

    Returns:
    numpy.ndarray: The weight vector of shape (n_features,).
    """
    n_features = X_i.shape[1]

    # Compute X_i^T X_i
    XtX = X_i.T @ X_i

    # Add regularization term: lambda * I
    regularization = lambd * np.eye(n_features)

    # Solve the linear system (X_i^T X_i + lambda * I) w_i = X_i^T y_i
    XtY = X_i.T @ y_i
    w_i = np.linalg.solve(XtX + regularization, XtY)

    return w_i

def ridge_regression_stable_batch(X, y, sigma2_w, sigma2_y, n_jobs=-1):
    """
    Perform ridge regression independently for each of N problems in parallel.

    Parameters:
    X (numpy.ndarray): Input features of shape (N, n_samples, n_features).
    y (numpy.ndarray): Target values of shape (N, n_samples).
    lambd (float): Regularization parameter (lambda).
    n_jobs (int): The number of parallel jobs to run. -1 means using all processors.

    Returns:
    numpy.ndarray: The weight matrix of shape (N, n_features).
    """
    N, _, _ = X.shape

    lambd = sigma2_y / sigma2_w

    # Use joblib to parallelize the loop
    w_list = Parallel(n_jobs=n_jobs)(delayed(ridge_single_problem)(X[i], y[i], lambd) for i in range(N))

    # Convert the list of weight vectors to an array
    w = np.array(w_list)

    return w

def compute_evidence(X, ys, w_MAP, sigma2_w, sigma2_y):
    n_repeats, n_samples, n_features = X.shape

    # alpha = 1 / sigma2_w
    alpha = 1 / sigma2_w
    beta = 1 / (sigma2_y)

    # constant terms
    const_term = 0.5 * n_features * np.log(alpha) + 0.5 * n_samples * np.log(beta) - 0.5 * n_samples * np.log(2 * np.pi)

    # Error term (Phi = samples x features)
    preds = (w_MAP[:, np.newaxis, :] * X).sum(-1)
    norm_squared = np.sum(w_MAP**2, axis=1)
    error_term = -0.5 * beta * np.sum((ys - preds) ** 2, axis=1) - 0.5 * alpha * norm_squared

    # Determinant term
    A = alpha * np.eye(n_features) + beta * np.matmul(np.transpose(X, (0, 2, 1)), X)
    log_det_term = -0.5 * np.linalg.slogdet(A)[1]

    log_evidence_batch = const_term + error_term + log_det_term

    log_likelihood =  0.5 * n_samples * np.log(beta) - 0.5 * n_samples * np.log(2 * np.pi) -0.5 * beta * np.sum((ys - preds) ** 2, axis=1)

    return log_evidence_batch, log_likelihood


def fourier_transform(xs_b, m):
    if len(xs_b.shape) == 2:
        cosine_features = np.cos((np.pi / L) * np.arange(m + 1)[np.newaxis, np.newaxis, :] * xs_b[:, :, np.newaxis]) 
        sine_features = np.sin((np.pi / L) * np.arange(1, m + 1)[np.newaxis, np.newaxis, :] * xs_b[:, :, np.newaxis]) 

        features_x = np.concatenate([cosine_features, sine_features], -1) 
    elif len(xs_b.shape) == 1:
        cosine_features = np.cos((np.pi / L) * np.arange(m + 1)[np.newaxis, :] * xs_b[:, np.newaxis]) 
        sine_features = np.sin((np.pi / L) * np.arange(1, m + 1)[np.newaxis, :] * xs_b[:, np.newaxis]) 

        features_x = np.concatenate([cosine_features, sine_features], -1) 

    return features_x

def sample_weights(num_repeats, m, sigma2_w):
    a_coeffs = np.random.normal(size=(num_repeats, m+1))
    b_coeffs = np.random.normal(size=(num_repeats, m))
    rand_w = np.concatenate([a_coeffs, b_coeffs], -1) / np.sqrt(2*m+1)
    return rand_w * np.sqrt(sigma2_w)

def compute_outputs(true_w, xs_b, sigma2_y, eps=None):
    m = int(true_w.shape[-1] / 2)
    features_x = fourier_transform(xs_b, m)
    true_val = (true_w[:, np.newaxis, :] * features_x).sum(-1)
    if eps is None:
        noise = np.random.randn(*true_val.shape) * np.sqrt(sigma2_y)
    else:
        noise = eps * np.sqrt(sigma2_y)

    return true_val, true_val + noise


def gather_ridge_weights_with_evidence(xs, ys, candidates, sigma2_w, sigma2_y, xs_test, is_ols=False):
    ridge_ws = []
    log_evidences = []
    log_likelihoods = []
    preds = []
    for candidate_dim in candidates:
        features_x = fourier_transform(xs, candidate_dim) / np.sqrt(2*candidate_dim + 1)

        if not is_ols: 
            ridge_w = ridge_regression_stable_batch(features_x, ys, sigma2_w, sigma2_y)
        else: 
            # 1e-6 is for numerical stability
            ridge_w = ridge_regression_stable_batch(features_x, ys, 1.0, 1e-6)
        log_evidence, log_likelihood = compute_evidence(features_x, ys, ridge_w, sigma2_w, sigma2_y)

        ridge_ws.append(ridge_w)
        log_evidences.append(log_evidence)
        log_likelihoods.append(log_likelihood)

        features_test = fourier_transform(xs_test, candidate_dim) / np.sqrt(2*candidate_dim + 1)
        pred = (features_test * ridge_w).sum(-1)

        preds.append(pred)

    return ridge_ws, np.array(log_evidences), np.array(preds), np.array(log_likelihoods)


def get_bayes_preds(xs, ys_true,
    icl_length, m, sigma2_w, sigma2_y, tail_idx, sigma2_w_data=None, sigma2_y_data=None, bmc_adj=False):
    num_repeats = xs.shape[0]
    compl_cands = [1 + i for i in range(10)]

    if sigma2_w_data is None:
        sigma2_w_data = sigma2_w
    if sigma2_y_data is None:
        sigma2_y_data = sigma2_y

    ## Generate x, y with respect to the true frequency
    # X = [batch, icl length]
    # y = [batch, icl length]
    # X_test = [batch]
    # y_test = [batch]
    xs_examples = xs[:, :-1]
    ys_examples = ys_true[:, :-1]

    xs_test = xs[:, -1]
    ys_test = ys_true[:, -1]

    ## Obtain predictions from ridge regression for specified model complexities under different sigma
    # for sigma_cand in sigma_cands:
    ws_true, evi_true, pred_true, ll_true = gather_ridge_weights_with_evidence(
        xs_examples, ys_examples, compl_cands, sigma2_w, sigma2_y, xs_test)

    ## Construct baseline predictions
    # Baseline 1: Bayesian model averaging
    frs = 1 / np.arange(1, 1 + 10)**tail_idx
    evi_adj = np.exp(evi_true[:10] - np.max(evi_true[:10])) * np.expand_dims(frs, -1)
    evi_adj = evi_adj / np.sum(evi_adj, axis=0, keepdims=True)
    pred_bma = (pred_true[:10] * evi_adj).sum(0)

    # Baseline 2: Oracle predictor
    # m starts from 1 so subtract it by 1 
    pred_oracle = pred_true[m-1]

    # Baseline 3: MAP predictor
    evi_adj = np.expand_dims(frs, -1) * evi_true[:10]
    pred_map = pred_true[np.argmax(evi_adj, 0), np.arange(num_repeats)]

    # Baselin 4: Model averaging with equal weights
    frs_adj = np.expand_dims(frs / frs.sum(), -1)
    naive_ensemble = (pred_true[:10] * frs_adj).sum(0)

    ## 
    num_params = np.expand_dims(np.array([1 + 2*npm for npm in range(1, 11)]), -1)
    num_samples = xs_examples.shape[-1]
    is_aicc_applicable = (- num_params - 2 + num_samples > 0)

    aic = ll_true - num_params
    aicc = aic + 0.5*(num_params + 1)*(num_params + 2) / (- num_params - 2 + num_samples)
    aicc = np.where(is_aicc_applicable, aicc, aic)
    bic = ll_true - 0.5*num_params*np.log(num_samples)

    # Baselines 5,6,7 => aic, aicc, bic
    pred_aic_ridge = pred_true[np.argmax(aic, 0), np.arange(num_repeats)]
    pred_aicc_ridge = pred_true[np.argmax(aicc, 0), np.arange(num_repeats)]
    pred_bic_ridge = pred_true[np.argmax(bic, 0), np.arange(num_repeats)]

    # Compare
    quant_of_interests = ([pred_bma, pred_oracle, pred_map, naive_ensemble,
                            pred_aic_ridge, pred_aicc_ridge, pred_bic_ridge])

    return quant_of_interests

def main(model_name, m, sigma2_w, sigma2_y, xs_og, eps_og, max_len, icl_freq, args):
    # Prepare some configurations
    tail_idx = 0.
    task = model_name

    run_dir = args.run_dir
    run_id = os.listdir(os.path.join(run_dir, task))
    if len(run_id) > 1:
        raise NotImplementedError
    run_id = run_id[0]
    saved_path = os.path.join(run_dir, task, run_id)

    fine_grained = 10
    icl_num = int(max_len / icl_freq)
    icl_start = int(fine_grained / icl_freq)
    icl_cands = list(range(1, fine_grained + 1)) + [1 + icl_freq*i for i in range(icl_start, icl_num)]
    icl_cands = np.array(icl_cands)
    steps = [1000000]

    ## Data generation
    true_w = sample_weights(num_repeats, m, sigma2_w)
    ys_true, ys_noise = compute_outputs(true_w, xs_og, sigma2_y, eps_og)

    ## Get transformer predictions 
    # trans_preds = [len, batch]
    with torch.no_grad():
        for step in steps:
            model, conf = get_model_from_run(saved_path, step=step)
            model = model.cuda()
            model.eval()

            ## Generate data
            xs_icl = torch.from_numpy(xs_og).unsqueeze(-1).cuda().float()
            ys_icl = torch.from_numpy(ys_noise).cuda().float()
            outs = model(xs_icl, ys_icl)
            print(f'Transformer prediction shape: {outs.shape}')

    trans_preds = outs.transpose(1, 0).cpu().numpy()[icl_cands]

    ## Get baseline predictions
    base_preds = []
    bmas = []
    for icl_idx, icl_length in enumerate(tqdm(icl_cands)):
        # +1 account for the test sample
        xs = xs_og[:, :icl_length+1] 
        eps = eps_og[:, :icl_length+1]
        ys_data = copy.deepcopy(ys_noise[:, :icl_length+1])

        base_pred = get_bayes_preds(xs, ys_data, icl_length, m, sigma2_w, sigma2_y, tail_idx)
        base_pred = np.array(base_pred)
        
        if icl_idx == 0:
            print(f'base_pred shape: {base_pred[0].shape}')
        base_preds.append(base_pred)

    base_preds = np.array(base_preds)
       
    return trans_preds, base_preds, ys_noise, ys_true


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir

    sigma_cands = [
        [0.1,  0.1], [0.1,  1], 
        [1.0, 0.1], [1.0, 1], [1.0, 10.],
        [10., 0.1], [10., 1], [10., 10.]
        ]
    sigma2_y_base = 0.25/8.

    ms = list(range(1,11))

    max_len = 256
    icl_freq = 3
    num_repeats = 512

    ## Generate samples and noises 
    xs = np.random.uniform(low=-5, high=5, size=(num_repeats, max_len)) 
    np.save(f'{output_dir}/xs_bm2', xs)
    eps = np.random.randn(num_repeats, max_len)
    np.save(f'{output_dir}/eps_bm2', eps)

    for m in ms:
        print('+'*100)
        print('+'*100)
        print('\t\tPerforming for model ', m)
        print('+'*100)
        mses, dists = [], []
        predictions = []
        base_predictions = []
        targets = []
        true_targets = []
        for i_s, sigma_cand in enumerate(tqdm(sigma_cands)):
            sigma2_w = sigma_cand[0]
            sigma2_y = sigma2_y_base * sigma_cand[1]

            if sigma2_w == 0:
                sigma2_w_name = '0'
            elif sigma2_w == 10:
                sigma2_w_name = '10'
            elif sigma2_w == 0.1:
                sigma2_w_name = '01'
            elif sigma2_w == 1:
                sigma2_w_name = '1'
            else:
                sigma2_w_name = sigma2_w

            if sigma_cand[1] == 1:
                sigma2_y_name = '1'
            elif sigma_cand[1] == 10:
                sigma2_y_name = '10'
            elif sigma_cand[1] == 0.1:
                sigma2_y_name = '01'
            elif sigma_cand[1] == 0:
                sigma2_y_name = '0'
            else:
                sigma2_y_name = sigma_cand[1]

            model_name = 'w_{}_ep_{}'.format(sigma2_w_name, sigma2_y_name)

            prediction, base_prediction, ys_noise, ys_true = main(
                model_name, m, sigma2_w, sigma2_y, xs_og=xs, eps_og=eps,
                max_len=max_len, icl_freq=icl_freq, args=args)

            predictions.append(prediction)
            base_predictions.append(base_prediction)
            targets.append(ys_noise)
            true_targets.append(ys_true)

        np.save(f'{output_dir}/trans_{m}', predictions)
        np.save(f'{output_dir}/base_{m}', base_predictions)
        np.save(f'{output_dir}/ys_noise_{m}', targets)
        np.save(f'{output_dir}/ys_true_{m}', true_targets)


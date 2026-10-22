function [best_j, est_norm, gamma] = sparsified_1norm_est(M, k, TOL, idx_mode)
% Alg 2.1 - Sparsified LAPACK 1-Norm Estimator
% M: input matrix (n x n)
% k: sparsification parameter
% TOL: maximum iterations (>= 2)
    if nargin < 4 || isempty(idx_mode)
        idx_mode = 'random';
    end
    
    [n, ~] = size(M);
    
    % ---------------------------------------------------------------------
    % INITIALIZATION
    % ---------------------------------------------------------------------

    % Sliced Generation for Mg
    idx_g = pick_idx(n, k, idx_mode);
    Mg = sum(M(:, idx_g), 2) / k; 

    % Sliced Generation for Mh
    idx_h = pick_idx(n, k, idx_mode);
    i_minus_1 = (idx_h - 1)';
    h_vals = ((-1).^i_minus_1) .* (1 + i_minus_1 / (n - 1));
    h_k_vals = h_vals / norm(h_vals, 1); % The k non-zero values

    % Multiply only the k extracted columns of M by the k values of h
    Mh = M(:, idx_h) * h_k_vals;

    if norm(Mg, 1) >= norm(Mh, 1)
        u = Mg;
    else
        u = Mh;
    end
    
    % Initialize tracking variables
    gamma = 0;
    nu_prev = -1;
    j_prev = -1;
    
    % ---------------------------------------------------------------------
    % COMPUTATIONS
    % ---------------------------------------------------------------------
    while true

        % 1. Compute w, sparsify, and project x
        w = sign(u);
        w(w == 0) = 1; % Handle 0s natively like LAPACK
        
        idx_w = pick_idx(n, k, idx_mode);
        x = (M(idx_w, :))' * w(idx_w);

        gamma = gamma + 1;

        % 2. Find column index j yielding the maximum absolute value in x
        [~, j_curr] = max(abs(x));
        
        % 3. Update u and calculate current norm
        u = M(:, j_curr);
        nu_curr = norm(u, 1);
        
        % 4. Check for early stopping (local max reached)
        if nu_prev >= nu_curr
            best_j = j_prev;
            est_norm = nu_prev;
            break;
        end
        
        % 5. Check for tolerance stopping
        if gamma == TOL
            best_j = j_curr;
            est_norm = nu_curr;
            break;
        end
        
        % 6. Iterate
        nu_prev = nu_curr;
        j_prev = j_curr;
    end
end

% % =========================================================================
% % Helper Function: k-Sparsification
% % =========================================================================

function idx = pick_idx(n, k, mode)
% Choose k distinct indices from 1..n.
% 'random': uniformly at random, without replacement
% 'even':   fixed; the k+1 stretches of unused indices differ in
%           length by at most one (k = 1 gives n/2 for even n)
% 'offset': the even pattern shifted by a random amount, uniform over
%           all shifts that keep every index inside 1..n
    switch mode
        case 'random'
            idx = randperm(n, k);
        case 'even'
            idx = floor((1:k) * (n+1) / (k+1));
        case 'offset'
            p   = floor((1:k) * (n+1) / (k+1));
            idx = p + randi([1 - p(1), n - p(end)]);
        otherwise
            error('pick_idx: unknown mode ''%s''', mode);
    end
end

function [best_j, est_norm, gamma, mv_steps] = sparsified_1norm_est_w_ca(M, k, tol, TOL, idx_mode)
    % Alg 4.2 - Sparsified LAPACK 1-Norm Estimator with C-A steps.
    % M:        input matrix (n x n)
    % k:        sparsification parameter
    % tol:      maximum invocations of maxvol
    % TOL:      maximum iterations for the estimator
    %
    % CHANGE: now returns mv_steps, the TOTAL number of maxvol iterations
    % consumed across all invocations. Needed for the entries/flops cost
    % model -- each maxvol step reads one row and one column of M, so the
    % C-A stage is not free and its cost has to be reported.
    
    if nargin < 5 || isempty(idx_mode)
        idx_mode = 'random';
    end

    [n, ~] = size(M);

    % ---------------------------------------------------------------------
    % INITIALIZATION
    % ---------------------------------------------------------------------

    [~, j_prev, ~, mv_steps] = maxvol_1(M, [], idx_mode);
    u = M(:, j_prev);
    nu_prev = norm(u, 1);
    gamma    = 0;

    % ---------------------------------------------------------------------
    % COMPUTATIONS
    % ---------------------------------------------------------------------
    while true
        gamma = gamma + 1;

        w = sign(u);
        w(w == 0) = 1;

        idx_w = pick_idx(n, k, idx_mode);
        x = (M(idx_w, :))' * w(idx_w);

        [~, j_curr] = max(abs(x));

        u = M(:, j_curr);
        nu_curr = norm(u, 1);

        % Stage 5: early stopping
        if nu_prev >= nu_curr
            best_j   = j_prev;
            est_norm = nu_prev;
            break;
        end

        % Stage 6: tolerance stopping
        if gamma == TOL
            best_j   = j_curr;
            est_norm = nu_curr;
            break;
        end

        nu_prev = nu_curr;
        j_prev  = j_curr;
    end
end


function idx = pick_idx(n, k, mode)
% Choose k distinct indices from 1..n.
% 'random': uniformly at random, without replacement
% 'even':   fixed; the k+1 stretches of unused indices differ in
%           length by at most one (k = 1 gives n/2 for even n)
    switch mode
        case 'random'
            idx = randperm(n, k);
        case 'even'
            idx = floor((1:k) * (n+1) / (k+1));
        otherwise
            error('pick_idx: unknown mode ''%s''', mode);
    end
end
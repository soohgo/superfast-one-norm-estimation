function [best_j, est_norm, gamma, mv_steps] = sparsified_1norm_est_w_ca(M, k, tol, TOL, idx_mode)
    % Sparsified LAPACK 1-Norm Estimator with C-A steps.
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

        % In-line k-sparsification; equivalent to
        %   w_bar = k_sparsify(w, k);  x = M' * w_bar;
        idx_w = pick_idx(n, k, idx_mode);
        x = (M(idx_w, :))' * w(idx_w);

        [~, j_curr] = max(abs(x));

        u = M(:, j_curr);
        nu_curr = norm(u, 1);

        % early stopping
        if nu_prev >= nu_curr
            best_j   = j_prev;
            est_norm = nu_prev;
            break;
        end

        % tolerance stopping
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
            % Another choice for randomly offset grid
            % idx = floor(((0:k-1) + rand) * n / k) + 1;
        case 'mixed'
            fixed_idx = floor((1:k) * (n+1) / (k+1));
            rest = setdiff(1:n, fixed_idx);                 % indices not yet used
            idx  = [fixed_idx, rest(randperm(numel(rest), k))];
        otherwise
            error('pick_idx: unknown mode ''%s''', mode);
    end
end

function [best_j, est_norm, gamma, mv_steps] = sparsified_1norm_est_w_ca_hybrid(M, k, tol, TOL)
    % Sparsified LAPACK 1-Norm Estimator with C-A steps, using mixed mode
    % M:        input matrix (n x n)
    % k:        sparsification parameter
    % tol:      maximum invocations of maxvol
    % TOL:      maximum iterations for the estimator
    %
    % CHANGE: now returns mv_steps, the TOTAL number of maxvol iterations
    % consumed across all invocations. Needed for the entries/flops cost
    % model -- each maxvol step reads one row and one column of M, so the
    % C-A stage is not free and its cost has to be reported.

    [n, ~] = size(M);

    % ---------------------------------------------------------------------
    % INITIALIZATION
    % ---------------------------------------------------------------------

    [~, j_prev_g, ~, mv_steps] = maxvol_1(M, [], 'random');
    Mg = M(:, j_prev_g);
    
    [~, j_prev_h, ~, mv_steps] = maxvol_1(M, [], 'even');
    Mh = M(:, j_prev_h);

    if norm(Mg, 1) >= norm(Mh, 1)
        u = Mg;
        j_prev = j_prev_g;
    else
        u = Mh;
        j_prev = j_prev_h;
    end

    nu_prev = norm(u, 1);
    gamma    = 0;

    % ---------------------------------------------------------------------
    % COMPUTATIONS
    % ---------------------------------------------------------------------

    while true
        gamma = gamma + 1;

        w = sign(u);
        w(w == 0) = 1;

        idx_w = pick_idx(n, k, 'random');
        x = (M(idx_w, :))' * w(idx_w);
        [~, j_curr] = max(abs(x));
        u = M(:, j_curr);
        nu_curr = norm(u, 1);

        idx_w_e = pick_idx(n, k, 'even');
        x_e = (M(idx_w_e, :))' * w(idx_w_e);
        [~, j_curr_e] = max(abs(x_e));
        u_e = M(:, j_curr_e);
        nu_curr_e = norm(u_e, 1);


        if nu_curr_e > nu_curr
            j_curr = j_curr_e;
            u = u_e;
            nu_curr = nu_curr_e;
        end
        
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
        otherwise
            error('pick_idx: unknown mode ''%s''', mode);
    end
end

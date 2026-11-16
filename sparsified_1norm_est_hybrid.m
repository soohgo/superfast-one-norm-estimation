function [best_j, est_norm, gamma] = sparsified_1norm_est_hybrid(M, k, TOL)
    % Alg 3.1-H - Sparsified LAPACK 1-Norm Estimator (Hybrid)
    n = size(M, 1);

    % Precompute constant equidistant index set once
    idx_even = floor((1:k) * (n + 1) / (k + 1));

    % ---------------------------------------------------------------------
    % INITIALIZATION (Both g and h evaluated on both random and even modes)
    % ---------------------------------------------------------------------
    % 1. Random g and h (2k columns read)
    idx_g_rand = randperm(n, k);
    Mg_rand = sum(M(:, idx_g_rand), 2) / k;

    idx_h_rand = randperm(n, k);
    i_m1_r = (idx_h_rand - 1)';
    h_r = ((-1).^i_m1_r) .* (1 + i_m1_r / (n - 1));
    Mh_rand = M(:, idx_h_rand) * (h_r / norm(h_r, 1));

    % 2. Equispaced g and h (shared slice: only k columns read)
    M_even = M(:, idx_even);
    Mg_even = sum(M_even, 2) / k;

    i_m1_e = (idx_even - 1)';
    h_e = ((-1).^i_m1_e) .* (1 + i_m1_e / (n - 1));
    Mh_even = M_even * (h_e / norm(h_e, 1));

    % Pick the best initial vector u across all four candidates
    candidates = {Mg_rand, Mh_rand, Mg_even, Mh_even};
    norms = [norm(Mg_rand, 1), norm(Mh_rand, 1), norm(Mg_even, 1), norm(Mh_even, 1)];
    [~, best_init] = max(norms);
    u = candidates{best_init};

    % Force at least one column selection step
    nu_prev = -1;
    j_prev  = -1;
    gamma   = 0;

    % ---------------------------------------------------------------------
    % COMPUTATIONS
    % ---------------------------------------------------------------------
    while true
        gamma = gamma + 1;

        w = sign(u);
        w(w == 0) = 1;

        % 1. Random k-sparsification
        idx_rand = randperm(n, k);
        x_rand = abs(w(idx_rand)' * M(idx_rand, :));
        [~, j_curr] = max(x_rand);
        
        u = M(:, j_curr);
        nu_curr = norm(u, 1);

        % 2. Equispaced k-sparsification
        x_even = abs(w(idx_even)' * M(idx_even, :));
        [~, j_curr_e] = max(x_even);

        if j_curr_e ~= j_curr
            u_e = M(:, j_curr_e);
            nu_curr_e = norm(u_e, 1);
            if nu_curr_e > nu_curr
                j_curr = j_curr_e;
                u = u_e;
                nu_curr = nu_curr_e;
            end
        end

        % Early stopping
        if nu_prev >= nu_curr
            best_j   = j_prev;
            est_norm = nu_prev;
            break;
        end

        % Tolerance stopping
        if gamma == TOL
            best_j   = j_curr;
            est_norm = nu_curr;
            break;
        end

        nu_prev = nu_curr;
        j_prev  = j_curr;
    end
end
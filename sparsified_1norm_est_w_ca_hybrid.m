function [best_j, est_norm, gamma, mv_steps] = sparsified_1norm_est_w_ca_hybrid(M, k, tol, TOL)
% Alg 4.2-H - Sparsified LAPACK 1-Norm Estimator (Hybrid)
n = size(M, 1);

% ---------------------------------------------------------------------
% INITIALIZATION
% ---------------------------------------------------------------------
[~, j_prev_g, ~, mv_g] = maxvol_1(M, [], 'random');
[~, j_prev_h, ~, mv_h] = maxvol_1(M, [], 'even');
mv_steps = mv_g + mv_h;

Mg = M(:, j_prev_g);
nu_g = norm(Mg, 1);

if j_prev_g == j_prev_h
    u = Mg;
    j_prev = j_prev_g;
    nu_prev = nu_g;
else
    Mh = M(:, j_prev_h);
    nu_h = norm(Mh, 1);
    if nu_g >= nu_h
        u = Mg;
        j_prev = j_prev_g;
        nu_prev = nu_g;
    else
        u = Mh;
        j_prev = j_prev_h;
        nu_prev = nu_h;
    end
end

gamma = 0;
idx_even = floor((1:k) * (n + 1) / (k + 1));

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

    % 2. Equidistant k-sparsification
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
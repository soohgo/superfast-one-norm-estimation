%% run_benchmarks.m
% Full benchmark suite for superfast 1-norm estimation.
% ---------------------------------------------------------------------
% REQUIREMENTS
%   Anymatrix on the path. MatrixGenerator.m with haar_orth. And
%   sparsified_1norm_est_w_ca.m returning mv_steps as its fourth output.
%   No toolboxes: percentiles are computed locally.


clear; clc;
addpath('anymatrix');
outfilepath = 'output'
if ~exist(outfilepath,'dir'), mkdir(outfilepath); end

%% ------------------------------ CONFIG ------------------------------
QUICK = false;          % true = ~5 minute smoke test, not for the paper

if QUICK
    sizes = [256 512];  trials = [20 10];  ttrials = [3 3];
    matrix_types = {'Shaw','Fast Decay','Random', 'Kahan','Poisson'};
    time_target = 5e-3;
else
    sizes   = [1024, 2048, 4096];
    trials  = [1000,  1000,  1000];   % accuracy trials
    ttrials = [  100,   100,   100];   % of which these are also timed
    matrix_types = {'Shaw','Gravity','SLP','Fast Decay','Slow Decay', ...
                    'One Small SV','One Large SV','Cauchy','Random', ...
                    'Inv Gaussian', 'Kahan','Poisson'};
    time_target = 20e-3;            % seconds per timed batch
end

det_types = {'Shaw','Gravity','SLP','Kahan','Poisson'};  % generated once per dimension
k_funcs   = {@(n) 1, @(n) floor(log2(log2(n))), @(n) floor(log2(n))};
TOL       = 10;      % iteration cap for the sparsified estimators
tol_ca    = 1;       % maximum maxvol invocations inside Alg 4.2

num_M = numel(matrix_types);  num_S = numel(sizes);  num_K = numel(k_funcs);
nthreads = maxNumCompThreads;
stamp = char(datetime('now','Format','yyyyMMdd_HHmmss'));
diary(sprintf('%s/benchmark_log_%s.txt', outfilepath, stamp)); diary on;

%% --------------------------- ENVIRONMENT ----------------------------
env = sprintf('MATLAB: %s\nComputer: %s\nThreads: %d\nTimestamp: %s\nQUICK: %d\nSizes: %s\nTrials: %s\nTimed trials: %s\nTOL: %d\ntol_ca: %d\n', ...
    version, computer, nthreads, stamp, QUICK, mat2str(sizes), ...
    mat2str(trials), mat2str(ttrials), TOL, tol_ca);
try
    if ismac
        [~,c] = system('sysctl -n machdep.cpu.brand_string');
        [~,m] = system('sysctl -n hw.memsize');
        env = [env sprintf('CPU: %sRAM (bytes): %s', c, m)];
    elseif isunix
        [~,c] = system('grep -m1 "model name" /proc/cpuinfo | cut -d: -f2');
        [~,m] = system('grep MemTotal /proc/meminfo');
        env = [env sprintf('CPU:%sRAM: %s', c, m)];
    end
catch
end
fprintf('%s\n', env);
f = fopen(sprintf('%s/environment_%s.txt', outfilepath, stamp),'w'); fprintf(f,'%s',env); fclose(f);

%% ----------------------------- WARM-UP ------------------------------
fprintf('Warming up...\n');
Mw = randn(512);
for w = 1:5
    normest1(Mw); normest1(Mw,1); norm(Mw,1); maxvol_1(Mw);
    sparsified_1norm_est(Mw,1,TOL);
    sparsified_1norm_est_w_ca(Mw,1,tol_ca,TOL);
    maxvol_1(Mw,7);
    sparsified_1norm_est_w_ca_hybrid(Mw,1,tol_ca,TOL);
end
clear Mw w
nb = 1e6; ts = tic; for r = 1:nb, end; loop_ns = toc(ts)/nb*1e9;
fprintf('Bare loop overhead: %.2f ns/iteration', loop_ns);
if loop_ns > 100, fprintf('  <-- WARNING: timings may be inflated\n');
else,             fprintf('  (negligible)\n'); end

%% ------------------------------ STORAGE -----------------------------
% R{m,s} holds every per-trial vector for one (class, dimension) cell.
R = cell(num_M, num_S);

%% ------------------------------- MAIN -------------------------------
for m_idx = 1:num_M
    mtype  = matrix_types{m_idx};
    is_det = ismember(mtype, det_types);

    for s_idx = 1:num_S
        n = sizes(s_idx);  ntr = trials(s_idx);  ntt = ttrials(s_idx);
        tcell = tic;
        fprintf('\n[%s | n = %d] %d trials, %d of them timed\n', mtype, n, ntr, ntt);

        if is_det
            M_fixed = MatrixGenerator.get_matrix(mtype, n);
            tn_fixed = norm(M_fixed, 1);
            if size(M_fixed,1) ~= n
                fprintf(['  [note: %s returned order %d, not the requested %d. ' ...
                         'k and the cost model use the actual order.]\n'], ...
                        mtype, size(M_fixed,1), n);
            end
        end

        r = struct();
        r.err_lap1 = zeros(ntr,1);  r.prod_lap1 = zeros(ntr,1);
        r.err_lap2 = zeros(ntr,1);  r.prod_lap2 = zeros(ntr,1);
        r.err_a41_max  = zeros(ntr,1);
        r.err_a41  = zeros(ntr,1);  r.it_a41    = zeros(ntr,1);
        r.j41      = zeros(ntr,1);       % start column, reused when timing
        r.n_actual = n;
        r.err_a31  = zeros(ntr,num_K);  r.it_a31  = zeros(ntr,num_K);
        r.err_a31_even  = zeros(ntr,num_K);  r.it_a31_even  = zeros(ntr,num_K);
        r.err_a41k = zeros(ntr,num_K);  r.err_a41k_max = zeros(ntr,num_K); 
        r.it_a41k = zeros(ntr,num_K);
        r.err_a42  = zeros(ntr,num_K);  r.it_a42  = zeros(ntr,num_K);
        r.err_a42_even  = zeros(ntr,num_K);  r.it_a42_even  = zeros(ntr,num_K);
        r.err_a42_hybrid  = zeros(ntr,num_K);  r.it_a42_hybrid  = zeros(ntr,num_K);
        r.mv_a42   = zeros(ntr,num_K);
        r.t_norm = zeros(ntt,1);  r.t_lap1 = zeros(ntt,1);
        r.t_lap2 = zeros(ntt,1);  r.t_a41  = zeros(ntt,1);
        r.t_a31  = zeros(ntt,num_K);
        r.t_a31_even  = zeros(ntt,num_K);
        r.t_a41k = zeros(ntt,num_K);  
        r.t_a42  = zeros(ntt,num_K);
        r.t_a42_even  = zeros(ntt,num_K);
        r.t_a42_hybrid  = zeros(ntt,num_K);

        % Alg 4.1 draws its start column from a dedicated stream, so that it
        % does not depend on how many draws Algs 3.1 consumed first.
        s41 = RandStream('mt19937ar','Seed', 20260725);

        for t = 1:ntr
            rng(t);
            if is_det
                M = M_fixed;  tn = tn_fixed; tm = max(abs(M), [], 'all');
            else
                M = MatrixGenerator.get_matrix(mtype, n);
                tn = norm(M, 1);
                tm = max(abs(M), [], 'all');
            end

            % ---------------- accuracy ----------------

            % ------------------- baseline 
            [est,~,~,it] = normest1(M,1);
            r.err_lap1(t) = relerr(tn,est);  r.prod_lap1(t) = it(2);
            

            [est,~,~,it] = normest1(M);
            r.err_lap2(t) = relerr(tn,est);  r.prod_lap2(t) = it(2);
            

            % ------------------- our algs
            nn = size(M,1);  r.n_actual = nn;
            j41 = randi(s41, nn);  r.j41(t) = j41;
            [~,estj,est_max,g] = maxvol_1(M, j41);
            r.err_a41_max(t) = relerr(tm,est_max);
            est = norm(M(:,estj),1);
            r.err_a41(t) = relerr(tn,est);  r.it_a41(t) = g;
            
            j31s = zeros(1,num_K);
            for kk = 1:num_K
                k = max(1, k_funcs{kk}(nn));

                [j31,est,g] = sparsified_1norm_est(M,k,TOL,'random');
                r.err_a31(t,kk) = relerr(tn,est);  r.it_a31(t,kk) = g;
                j31s(kk) = j31;

                [~,est,g] = sparsified_1norm_est(M,k,TOL,'even');
                r.err_a31_even(t,kk) = relerr(tn,est);  r.it_a31_even(t,kk) = g;

                [~,estj,est,g] = maxvol_1(M, j31);
                r.err_a41k_max(t,kk) = relerr(tm,est);  
                est = norm(M(:,estj),1);
                r.err_a41k(t, kk) = relerr(tn,est); r.it_a41k(t,kk) = g;

                [~,est,g,gmv] = sparsified_1norm_est_w_ca(M,k,tol_ca,TOL,'random');
                r.err_a42(t,kk) = relerr(tn,est);  r.it_a42(t,kk) = g;
                r.mv_a42(t,kk) = gmv;

                [~,est,g,~] = sparsified_1norm_est_w_ca(M,k,tol_ca,TOL,'even');
                r.err_a42_even(t,kk) = relerr(tn,est);  r.it_a42_even(t,kk) = g;

                [~,est,g,~] = sparsified_1norm_est_w_ca_hybrid(M,k,tol_ca,TOL);
                r.err_a42_hybrid(t,kk) = relerr(tn,est);  r.it_a42_hybrid(t,kk) = g;
            end

            % ---------------- timing, same matrix ----------------
            if t <= ntt
                r.t_norm(t) = tcall(1,M,0,TOL,tol_ca,0,time_target);
                r.t_lap1(t) = tcall(2,M,0,TOL,tol_ca,0,time_target);
                r.t_lap2(t) = tcall(3,M,0,TOL,tol_ca,0,time_target);
                r.t_a41(t)  = tcall(8,M,0,TOL,tol_ca,j41,time_target);
                for kk = 1:num_K
                    k = max(1, k_funcs{kk}(nn));
                    r.t_a31(t,kk)  = tcall(5,M,k,TOL,tol_ca,0,time_target);
                    r.t_a42(t,kk)  = tcall(7,M,k,TOL,tol_ca,0,time_target);
                    r.t_a41k(t,kk) = r.t_a31(t,kk) + ...
                                     tcall(8,M,k,TOL,tol_ca,j31s(kk),time_target);
                    r.t_a42_even(t,kk)  = tcall(9,M,k,TOL,tol_ca,0,time_target);
                    r.t_a42_hybrid(t,kk)  = tcall(10,M,k,TOL,tol_ca,0,time_target);
                end
            end

            if mod(t, max(1,floor(ntr/10))) == 0
                el = toc(tcell);
                fprintf('  %d/%d  (%.0f s elapsed, ~%.0f s left)\n', t,ntr,el,el*(ntr-t)/t);
            end
        end

        R{m_idx,s_idx} = r;

        % ---------------- per-cell report ----------------
        nn = r.n_actual;
        for kk = 1:num_K
            k = max(1, k_funcs{kk}(nn));
            fprintf('\n  --- %s | n = %d | k = %d ---\n', mtype, nn, k);
            fprintf('  %-14s %10s %10s %9s %11s %9s\n', ...
                    'algorithm','mean err','median','mean its','median ms','vs t=1');
            %b = median(r.t_lap2);
            b = median(r.t_lap1);
            line('  norm(M,1)',    [],            [],            r.t_norm, b);
            line('  normest1 t=1', r.err_lap1,    [],            r.t_lap1, b);
            line('  normest1 t=2', r.err_lap2,    [],            r.t_lap2, b);
            line('  Alg 3.1',      r.err_a31(:,kk),  r.it_a31(:,kk),  r.t_a31(:,kk),  b);
            line('  Alg 3.1 e',      r.err_a31_even(:,kk),  r.it_a31_even(:,kk),[],  b);
            line('  Alg 4.1',      r.err_a41,        r.it_a41,        r.t_a41,        b);
            line('  Alg 4.1-k',    r.err_a41k(:,kk), r.it_a41k(:,kk), r.t_a41k(:,kk), b);
            line('  Alg 4.2',      r.err_a42(:,kk),  r.it_a42(:,kk),  r.t_a42(:,kk),  b);
            line('  Alg 4.2 e',      r.err_a42_even(:,kk),  r.it_a42_even(:,kk),  [],  b);
            line('  Alg 4.2 h',      r.err_a42_hybrid(:,kk),  r.it_a42_hybrid(:,kk), r.t_a42_hybrid(:,kk),  b);
        end
        fprintf('  [cell done in %.0f s]\n', toc(tcell));
        save(sprintf('%s/results_%s.mat', outfilepath, stamp), 'R', 'matrix_types', ...
             'sizes', 'trials', 'ttrials', '-v7.3');
    end
end

%% ------------------------------ EXPORT ------------------------------
fprintf('\nWriting CSV summaries...\n');
fs = fopen(sprintf('%s/summary_%s_full.csv', outfilepath, stamp),'w');
fprintf(fs,'alg,matrix,n,k,threads,quantity,n_obs,min,q1,median,q3,max,mean,std,p95,p99\n');
nzero = 0;  th = sprintf('%d', nthreads);

for m_idx = 1:num_M
    mt = matrix_types{m_idx};
    for s_idx = 1:num_S
        r = R{m_idx,s_idx};
        if isempty(r), continue; end
        n = r.n_actual;

        emit(fs,fo,'normest1_t1',mt,n,0,th,'err',    r.err_lap1);
        emit(fs,fo,'normest1_t2',mt,n,0,th,'err',    r.err_lap2);
        emit(fs,fo,'Alg41',      mt,n,0,th,'err',    r.err_a41);
        emit(fs,fo,'Alg41',      mt,n,0,th,'iters',  r.it_a41);
        nzero = nzero + emit(fs,fo,'exact_norm', mt,n,0,th,'time_ms',r.t_norm*1000);
        nzero = nzero + emit(fs,fo,'normest1_t1',mt,n,0,th,'time_ms',r.t_lap1*1000);
        nzero = nzero + emit(fs,fo,'normest1_t2',mt,n,0,th,'time_ms',r.t_lap2*1000);
        nzero = nzero + emit(fs,fo,'Alg41',      mt,n,0,th,'time_ms',r.t_a41*1000);

        for kk = 1:num_K
            k = max(1, k_funcs{kk}(n));
            emit(fs,fo,'Alg31', mt,n,k,th,'err',  r.err_a31(:,kk));
            emit(fs,fo,'Alg31e', mt,n,k,th,'err',  r.err_a31_even(:,kk));
            emit(fs,fo,'Alg41k',mt,n,k,th,'err',  r.err_a41k(:,kk));
            emit(fs,fo,'Alg42', mt,n,k,th,'err',  r.err_a42(:,kk));
            emit(fs,fo,'Alg42e', mt,n,k,th,'err',  r.err_a42_even(:,kk));
            emit(fs,fo,'Alg31', mt,n,k,th,'iters',r.it_a31(:,kk));
            emit(fs,fo,'Alg41k',mt,n,k,th,'iters',r.it_a41k(:,kk));
            emit(fs,fo,'Alg42', mt,n,k,th,'iters',r.it_a42(:,kk));
            nzero = nzero + emit(fs,fo,'Alg31', mt,n,k,th,'time_ms',r.t_a31(:,kk)*1000);
            nzero = nzero + emit(fs,fo,'Alg31e', mt,n,k,th,'time_ms',r.t_a31_even(:,kk)*1000);
            nzero = nzero + emit(fs,fo,'Alg41k',mt,n,k,th,'time_ms',r.t_a41k(:,kk)*1000);
            nzero = nzero + emit(fs,fo,'Alg42', mt,n,k,th,'time_ms',r.t_a42(:,kk)*1000);
            nzero = nzero + emit(fs,fo,'Alg42e', mt,n,k,th,'time_ms',r.t_a42_even(:,kk)*1000);
        end
    end
end
fclose(fs);

%% ------------------------------ CHECKS ------------------------------
fprintf('\n========================= CHECKS =========================\n');
if nzero == 0
    fprintf('PASS: no zero medians in the timing data.\n');
else
    fprintf('*** WARNING: %d timing rows have a zero median. ***\n', nzero);
end

% normest1 reads the whole matrix prod times per estimate and norm(M,1) reads
% it once, so normest1 cannot be much faster than prod times norm. If it is,
% the matrix was in cache and the timing understates the real cost. Needs
% nothing about the machine, only the pass counts.
fprintf('\nCache residency check:\n');
ncache = 0;
for m_idx = 1:num_M
    for s_idx = 1:num_S
        r = R{m_idx,s_idx};
        if isempty(r), continue; end
        tn_ = median(r.t_norm);  tl_ = median(r.t_lap2);  pr_ = mean(r.prod_lap2);
        expect = 0.8 * pr_ * tn_;
        if tl_ < expect
            ncache = ncache + 1;
            fprintf(['  n=%5d %-14s normest1 t=2 took %8.3f ms, expected at least ' ...
                     '%8.3f ms (%.1f passes, norm takes %7.3f ms)\n'], ...
                    r.n_actual, matrix_types{m_idx}, tl_*1000, expect*1000, pr_, tn_*1000);
        end
    end
end
if ncache == 0
    fprintf('PASS: no timing implies the matrix was cache resident.\n');
else
    fprintf(['*** %d cells failed. At those sizes the matrix fits in cache, so\n' ...
             '    normest1 is measured faster than it really is and the speedup\n' ...
             '    is understated. Report speedups only from sizes that pass. ***\n'], ncache);
end
fprintf('Wrote %s/summary_%s.csv, outliers_%s.csv', outfilepath, stamp,stamp);
diary off;

%% ========================= LOCAL FUNCTIONS ==========================

function e = relerr(tn, est)
% |||M||_1 - est|/||M||_1.
    e = abs(tn - est) / tn;
end

function t = tcall(code, M, k, TOL, tol_ca, j0, target)
% Grow the batch until it lasts `target` seconds, then repeat it and keep
% the SMALLEST elapsed time: interference from the scheduler or from other
% processes can only add time to a measurement, never subtract it, so the
% minimum is the least contaminated estimate. Without this a single
% preempted batch inflates the reported interquartile range out of all
% proportion.
    NBATCH = 3;  nrep = 1;
    while true
        el = run_batch(code,M,k,TOL,tol_ca,j0,nrep);
        if el >= target || nrep >= 1e7, break; end
        nrep = max(2*nrep, ceil(nrep*target/max(el,1e-9)));
    end
    best = el;
    for b = 2:NBATCH
        best = min(best, run_batch(code,M,k,TOL,tol_ca,j0,nrep));
    end
    t = best/nrep;
end

function el = run_batch(code, M, k, TOL, tol_ca, j0, nrep)
% One timed batch. The switch is evaluated once per batch and the routine
% is called directly inside the loop, so no per-call dispatch cost is
% included and nothing has to be subtracted from the result.
    n = size(M,1);
    ts = tic;
    switch code
        case 1, for r=1:nrep, norm(M,1);                                   end
        case 2, for r=1:nrep, normest1(M,1);                               end
        case 3, for r=1:nrep, normest1(M);                                 end
        case 4, for r=1:nrep, maxvol_1(M);                                 end
        case 5, for r=1:nrep, sparsified_1norm_est(M,k,TOL);               end
        case 7, for r=1:nrep, sparsified_1norm_est_w_ca(M,k,tol_ca,TOL);   end
        case 8, for r=1:nrep, maxvol_1(M,j0);                              end
        case 9, for r=1:nrep, sparsified_1norm_est_w_ca(M,k,tol_ca,TOL, 'even');   end
        case 10, for r=1:nrep, sparsified_1norm_est_w_ca_hybrid(M,k,tol_ca,TOL);   end
    end
    el = toc(ts);
end

function line(name, e, its, tt, base)
s1 = '         -';  s2 = '         -';  s3 = '        -';
s4 = '          -';  s5 = '        -';
if ~isempty(e),   s1 = sprintf('%10.4f', mean(e));   s2 = sprintf('%10.4f', median(e)); end
if ~isempty(its), s3 = sprintf('%9.2f', mean(its));  end
if ~isempty(tt),  s4 = sprintf('%11.4f', median(tt)*1000);
    s5 = sprintf('%8.1fx', base/median(tt)); end
fprintf('  %-14s %s %s %s %s %s\n', name, s1, s2, s3, s4, s5);
end


function z = emit(fs, fo, alg, mt, n, k, threads, qty, v)
% Summary statistics, plus the boxplot outliers (beyond 1.5 IQR) so that
% the figures can be redrawn faithfully without shipping every trial.
    z = 0;
    if isempty(v), return; end
    v = v(:);  q1 = pct(v,25);  q3 = pct(v,75);
    fprintf(fs,'%s,%s,%d,%d,%s,%s,%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n', ...
        alg,mt,n,k,threads,qty,numel(v),min(v),q1,median(v),q3,max(v), ...
        mean(v),std(v),pct(v,95),pct(v,99));
    if strcmp(qty,'time_ms')
        z = double(median(v) == 0);
    else
        w = 1.5*(q3-q1);  out = v(v < q1-w | v > q3+w);
        for o = 1:numel(out)
            fprintf(fo,'%s,%s,%d,%d,%s,%.6g\n',alg,mt,n,k,qty,out(o));
        end
    end
end

function v = pct(x,p)
% prctile without the Statistics Toolbox.
    x = sort(x(:));  m = numel(x);
    if m==0, v = NaN; return; end
    if m==1, v = x;   return; end
    q = ((0.5:1:(m-0.5))/m)*100;
    if p<=q(1),   v = x(1);   return; end
    if p>=q(end), v = x(end); return; end
    v = interp1(q,x,p,'linear');
end

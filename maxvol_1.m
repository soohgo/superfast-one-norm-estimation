function [i, j, est, iter] = maxvol_1(M, j_init, idx_mode)
% Alg 4.1 - Maxvol 1
if nargin < 3 || isempty(idx_mode)
    idx_mode = 'random';
end
if nargin < 2 || isempty(j_init)
    if strcmp(idx_mode, 'even')
        j = floor((size(M, 2) + 1) / 2);
    else
        j = randi(size(M, 2));
    end
else
    j = j_init;
end

iter = 0;
[est, i] = max(abs(M(:, j)));

while true
    iter = iter + 1;

    % Search in row i
    [row_max, j_new] = max(abs(M(i, :)));
    if row_max <= est
        break;
    end
    j = j_new;

    % Search in column j
    [col_max, i_new] = max(abs(M(:, j)));
    if col_max <= row_max
        est = row_max;
        break;
    end
    i = i_new;
    est = col_max;
end
end
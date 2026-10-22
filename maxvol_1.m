function [i, j, est, iter] = maxvol_1(M, j_init,idx_mode)
    % MaxVol 1: Computes the rank-1 maxvol Cross-Approximation.
    % M: input matrix (m x n)
    % j_init (optional): initial column index
    % idx_mode (optional): 'random', 'offset', 'mixed', or 'even'.
    %     random and offset do essentially the same thing, even picks index n/2
    
    % If no initial column is provided, pick one at random
    if nargin < 3 || isempty(idx_mode)
        idx_mode = 'random';
    end
    if nargin < 2 || isempty(j_init)
        j_init = pick_idx(size(M, 2), 1, idx_mode);
    end

    
    j = j_init;
    iter = 0; % Initialize iteration counter
    
    % Initialization: find the argmax i
    [~, i] = max(abs(M(:, j)));
    
    while true
        iter = iter + 1;

        % Find the argmax j_new in row i
        [max_row_val, j_new] = max(abs(M(i, :)));
        
        % If the max in the row is the same as our current element, we found a local max
        if max_row_val == abs(M(i, j))
            break;
        end
        j = j_new;
        
        % Find the argmax i_new  in column j
        [max_col_val, i_new] = max(abs(M(:, j)));
        
        % If the max in the column is the same as our current element, stop
        if max_col_val == abs(M(i, j))
            break;
        end
        i = i_new;
    end

    % Return the absolute value of the (i,j)-entry
    est = abs(M(i, j));
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
        case {'even', 'mixed'}
            idx = floor((1:k) * (n+1) / (k+1));
        case 'offset'
            p   = floor((1:k) * (n+1) / (k+1));
            idx = p + randi([1 - p(1), n - p(end)]);
        otherwise
            error('pick_idx: unknown mode ''%s''', mode);
    end
end

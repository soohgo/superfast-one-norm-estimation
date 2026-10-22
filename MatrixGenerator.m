classdef MatrixGenerator
    % MATRIXGENERATOR A collection of static methods to generate
    % test matrices for the 1-norm estimation benchmarks.

    methods (Static)

        function M = get_matrix(m_type, n)
            % Master router for generating any matrix by string name
            switch m_type
                case 'Shaw'
                    M = anymatrix('regtools/shaw', n);      % needs even n
                case 'Gravity'
                    M = anymatrix('regtools/gravity', n);
                case 'SLP'
                    M = MatrixGenerator.get_integral_equation_matrix(n);
                case 'Fast Decay'
                    M = MatrixGenerator.get_fast_decay_matrix(n);
                case 'Slow Decay'
                    M = MatrixGenerator.get_slow_decay_matrix(n);
                case 'One Small SV'
                    M = MatrixGenerator.get_one_small_sv_matrix(n);
                case 'One Large SV'
                    M = MatrixGenerator.get_one_large_sv_matrix(n);
                case 'Cauchy'
                    M = MatrixGenerator.get_cauchy_matrix(n);
                case 'Random'
                    M = MatrixGenerator.get_random_matrix(n);
                case 'Inv Gaussian'
                    M = MatrixGenerator.get_inv_gaussian_matrix(n);
                case 'Kahan'
                    M = MatrixGenerator.get_kahan_matrix(n);
                case 'Poisson'
                    M = MatrixGenerator.get_poisson_matrix(n);
                otherwise
                    error('Unknown matrix type specified: %s', m_type);
            end
        end

        function Q = haar_orth(n)
            % Uniformly (Haar) distributed orthogonal matrix on O(n).
            [Q, R] = qr(randn(n));
            d = sign(diag(R));
            d(d == 0) = 1;
            Q = Q .* d.';          % flip column signs
        end

        function M = get_cauchy_matrix(n)
            a = 0; b = 100; c = 100; d = 200;
            x = a + (b - a) * rand(n, 1);
            y = c + (d - c) * rand(1, n);
            M = 1 ./ (x - y);
        end

        function M = get_fast_decay_matrix(n, r)
            if nargin < 2, r = 20; end
            U = MatrixGenerator.haar_orth(n);
            V = MatrixGenerator.haar_orth(n);
            % first r values are 1, then fast decay, zero from the 101st on
            sig = [ones(r,1); (0.5).^(1:max(0, min(n,100) - r))'; zeros(max(0, n-100), 1)];
            M = (U .* sig.') * V';
        end

        function M = get_slow_decay_matrix(n, r)
            if nargin < 2, r = 20; end
            U = MatrixGenerator.haar_orth(n);
            V = MatrixGenerator.haar_orth(n);
            sig = [ones(r,1); ((2 : max(1, n-r+1)).^(-2))'];
            M = (U .* sig.') * V';
        end

        function M = get_one_small_sv_matrix(n)
            U = MatrixGenerator.haar_orth(n);
            V = MatrixGenerator.haar_orth(n);
            sig = ones(n,1);
            sig(end) = 10^-16 + (10^-3 - 10^-16) * rand();
            M = (U .* sig.') * V';
        end

        function M = get_one_large_sv_matrix(n)
            U = MatrixGenerator.haar_orth(n);
            V = MatrixGenerator.haar_orth(n);
            sig = ones(n,1);
            sig(1) = 10^3 + (10^16 - 10^3) * rand();
            M = (U .* sig.') * V';
        end

        function M = get_random_matrix(n)
            vals = [-1, 0, 1];
            M = vals(randi(3, n, n));
        end

        function M = get_integral_equation_matrix(n)
            t = (0:n-1)' * 2 * pi / n;
            r  = sqrt(2.5 + cos(3 * t));
            Ax = cos(t) .* r;
            Ay = r .* sin(t);
            Bx = 3 * cos(t);
            By = 3 * sin(t);
            ut = sqrt(2.5 + cos(3*t)) .* (-sin(t)) - cos(t) .* 3 .* sin(3*t) .* (2.5 + cos(3*t)).^(-0.5);
            vt = sqrt(2.5 + cos(3*t)) .* ( cos(t)) - sin(t) .* 3 .* sin(3*t) .* (2.5 + cos(3*t)).^(-0.5);
            dx = Bx - Ax';
            dy = By - Ay';
            log_term = 0.5 * log(dx.^2 + dy.^2);
            uv_term  = sqrt(ut'.^2 + vt'.^2);
            M = log_term .* uv_term;
        end

        function M = get_inv_gaussian_matrix(n)
            % Higham & Tisseur (2000) stress test for local maxima
            M = inv(randn(n));
        end

        function M = get_kahan_matrix(n)
            % Kahan's matrix (Higham's Test Matrix Toolbox), upper triangular.
            M = gallery('kahan', n);
        end

        function M = get_poisson_matrix(n)
            % 5-point discrete Laplacian: sparse, symmetric, and diagonally dominant
            k = round(sqrt(n));
            M = full(gallery('poisson', k));
        end

        function M = get_rectangular_matrix(m, n)
            q = min(m, n);
            r = min(20, q);
            U = MatrixGenerator.haar_orth(m);  U = U(:, 1:q);
            V = MatrixGenerator.haar_orth(n);  V = V(:, 1:q);
            sig = [ones(r,1); (0.5).^(1:max(0, min(q,100) - r))'; zeros(max(0, q-100), 1)];
            M = (U .* sig.') * V';
        end

    end
end

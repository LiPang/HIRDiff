function [Q, R, p] = sRRQR_rank(A, f, k)

%
%   Strong Rank Revealing QR with fixed rank 'k'
%
%       A(:, p) = Q * R = Q [R11, R12; 
%                              0, R22]
%   where R11 and R12 satisfies that matrix (inv(R11) * R12) has entries
%   bounded by a pre-specified constant which should be not less than 1. 
%   
%   Input: 
%       A, matrix,  target matrix that is appoximated.
%       f, scalar,  constant that bound the entries of calculated (inv(R11) * R12)%    
%       k, integer, dimension of R11. 
%
%   Output: 
%       A(:, p) = [Q1, Q2] * [R11, R12; 
%                               0, R22]
%               approx Q1 * [R11, R12];
%       Only truncated QR decomposition is returned as 
%           Q = Q1, 
%           R = [R11, R12];
%       where Q is a m * k matrix and R is a k * n matrix
%   
%   Reference: 
%       Gu, Ming, and Stanley C. Eisenstat. "Efficient algorithms for 
%       computing a strong rank-revealing QR factorization." SIAM Journal 
%       on Scientific Computing 17.4 (1996): 848-869.
%
%   Note: 
%       Algorithm 4 in the above ref. is implemented.




%   check constant bound f
if f < 1
    fprintf('parameter f given is less than 1. Automatically set f = 2\n');
    f = 2;
end

%   dimension of the given matrix
[m, n] = size(A);

%   modify rank k if necessary
k = min([k, m, n]);
    
%   pivoting QR first (generally most time consuming step)
[Q, R, p] = qr(A, 0);

%   check special case : rank equals the number of columns.
if (k == n)
    %   no need for SRRQR. 
    return ;
end


%   The following codes are the major part of the strong rank-revealing
%   algorithm which is based on the above pivoting QR. 
%   Name of variables are from the reference paper. 

%   make diagonals of R positive.
if (size(R,1) == 1 || size(R,2) == 1)
    ss = sign(R(1,1));
else
    ss = sign(diag(R));
end
R = bsxfun(@times, R, reshape(ss, [], 1));
Q = bsxfun(@times, Q, reshape(ss, 1, []));

%   Initialization of A^{-1}B ( A refers to R11, B refers to R12)
AB = linsolve(R(1:k, 1:k), R(1:k, (k+1):end), struct('UT', true)); 

%   Initialization of gamma, i.e., norm of C's columns (C refers to R22)
gamma = zeros(n-k, 1);                                    
if k ~= size(R,1)
    gamma = (sum(R((k+1):end, (k+1):end).^2, 1).^(1/2))';
end

%   Initialization of omega, i.e., reciprocal of inv(A)'s row norm 
tmp = linsolve(R(1:k, 1:k), eye(k), struct('UT', true));
omega = sum(tmp.^2, 2).^(-1/2);                          


%   KEY STEP in Strong RRQR: 
%   "while" loop for interchanging the columns from first k columns and 
%   the remaining (n-k) columns.


[Rm, ~] = size(R);
while 1
    %   identify interchanging columns
    tmp = (1./omega * gamma').^2 + AB.^2;
    [i,j] = find(tmp > f*f, 1, 'first');
    
    %   if no entry of tmp greater than f*f, no interchange needed and the
    %   present Q, R, p is a strong RRQR. 
    if isempty(i)           
        break;
    end
    
%     fprintf('interchanging\n');
    
    %   Interchange the i th and (k+j) th column of target matrix A and 
    %   update QR decomposition (Q, R, p), AB, gamma, and omega.
    %%   First step : interchanging the k+1 and k+j th columns    
    if j > 1  
        AB(:, [1, j]) = AB(:, [j, 1]);
        gamma([1, j]) = gamma([j, 1]);
        R(:, [k+1, k+j]) =R(:, [k+j, k+1]);
        p([k+1, k+j]) = p([k+j, k+1]);
    end
    
    %%   Second step : interchanging the i and k th columns
    if i < k
        p(i:k)     =  p([(i+1):k, i]);
        R(:, i:k)  =  R(:, [(i+1):k, i]);
        omega(i:k) =  omega([(i+1):k, i]);
        AB(i:k, :) =  AB([(i+1):k, i], :);
        %   givens rotation for the triangulation of R(1:k, 1:k)
        for ii = i : (k-1)
            G = givens(R(ii, ii), R(ii+1, ii));
            if G(1, :) * [R(ii,ii); R(ii+1,ii)] < 0
                G = -G;  %  guarantee R(ii,ii) > 0
            end
            R(ii:ii+1, :) = G * R(ii:ii+1, :);
            Q(:, ii:ii+1) = Q(:, ii:ii+1) * G';
        end
        if R(k,k) < 0
            R(k, :) = - R(k, :);
            Q(:, k) = -Q(:, k);
        end
    end

    %%   Third step : zeroing out the below-diag of k+1 th columns
    if k < Rm
        for ii = (k+2) : Rm
            G = givens(R(k+1, k+1), R(ii, k+1));
            if G(1, :) * [R(k+1, k+1); R(ii, k+1)] < 0
                G = -G;     %  guarantee R(k+1,k+1) > 0
            end 
            R([k+1, ii], :) = G * R([k+1, ii], :);
            Q(:,[k+1, ii]) = Q(:, [k+1, ii]) * G';
        end
    end

    %%   Fourth step : interchaing the k and k+1 th columns
    p([k,k+1]) = p([k+1,k]);
    ga = R(k, k);
    mu = R(k, k+1) / ga;         
    if k < Rm
        nu = R(k+1, k+1) / ga;
    else
        nu = 0;
    end
    rho = sqrt(mu*mu + nu*nu);
    ga_bar = ga * rho;
    b1 = R(1:(k-1), k);
    b2 = R(1:(k-1), k+1);
    c1T = R(k, (k+2):end);
    if (k+1 > Rm)
        c2T = zeros(1, length(c1T));
    else
        c2T = R(k+1, (k+2):end);
    end
    c1T_bar = (mu * c1T + nu * c2T)/rho;
    c2T_bar = (nu * c1T - mu * c2T)/rho;

    %   modify R
    R(1:(k-1), k) = b2;
    R(1:(k-1), k+1) = b1;
    R(k,k)     = ga_bar;
    R(k,k+1)   = ga * mu / rho;
    R(k+1,k+1) = ga * nu / rho;
    R(k, (k+2):end)   = c1T_bar;
    R(k+1, (k+2):end) = c2T_bar;

    %   update AB
    u = linsolve(R(1:k-1, 1:k-1), b1, struct('UT', true));
    u1 = AB(1:k-1, 1);
    AB(1:k-1, 1) = (nu*nu*u - mu*u1)/rho/rho;
    AB(k, 1) = mu / rho / rho;
    AB(k, 2:end) = c1T_bar / ga_bar;
    AB(1:k-1, 2:end) = AB(1:k-1, 2:end) + (nu*u*c2T_bar - u1*c1T_bar)/ga_bar;

    %   update gamma
    gamma(1) = ga * nu / rho;
    gamma(2:end) = (gamma(2:end).^2 + (c2T_bar').^2 - (c2T').^2).^(1/2);

    %   update omega
    u_bar = u1 + mu * u;
    omega(k) = ga_bar;
    omega(1:k-1) = (omega(1:k-1).^(-2) + u_bar.^2/(ga_bar*ga_bar) - u.^2/(ga*ga)).^(-1/2);

    %%   Eliminate new R(k+1, k) by orthgonal transformation           
    Gk = [mu/rho, nu/rho; nu/rho, -mu/rho];
    if k < Rm
        Q(:, [k,k+1]) = Q(:, [k,k+1]) * Gk';
    end                         
end

%   Only return the truncated version of the strong RRQR decomposition
Q = Q(:, 1:k);
R = R(1:k, :);
end

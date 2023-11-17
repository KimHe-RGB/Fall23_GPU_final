
m = 20;
n = 20;

% column major storage

v = zeros(m*n);

matrix_size = m*n*m*n

A = zeros(matrix_size,1);

for i = 1:m*n
    index_curr = (i-1) * m*n + i;

    index_next = min(index_curr + 1, matrix_size);
    index_prev = max(index_curr - 1, 1);
    A(index_next) = -1;
    A(index_prev) = -1;

    index_up = max(index_curr - n, 1);
    index_down = min(index_curr + n, matrix_size);
    if (index_up > (i-1)*m*n) 
        A(index_up) = -1; 
    end
    if (index_down < (i)*m*n) 
        A(index_down) = -1; 
    end
    
    A(index_curr) = 4;
end

A = reshape(A, m*n, m*n);


try 
    R = chol(A); % A = R' * R
    spy(R)
    % nnz(L)/matrix_size ~ O(1/n)
    disp('Matrix is symmetric positive definite.')
catch ME
    disp('Matrix is not symmetric positive definite')
end

% for i = 1:m*n
%     for j = 1:m*n
%         A(i * m*n + j)
%     end
% end
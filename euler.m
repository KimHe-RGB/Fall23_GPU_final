m = 100;
n = 100;

h = 0.1;
tau = 0.01;

% A: left side matrix 
A = zeros(m*n);
th = tau/h/h; % t*invhsq
for i = 1:m
    % center block
    for j = 1:n
        index = (i-1)*n+j;
        A(index, index) = 4*th;
        if j > 1
            A(index, index-1) = -th;
        end
        if j < n
            A(index, index+1) = -th;
        end
    end
    % left block
    if i > 1
        for j = 1:n
            index_x = (i-1)*n+j;
            index_y =  (i-2)*n+j;
            A(index_x, index_y) = -th;
        end
    end
    % rLight block
    if i < m
        for j = 1:n
            index_x = (i-1)*n+j;
            index_y =  i*n+j;
            A(index_x, index_y) = -th;
        end
    end
end

[L, D] = ldl(A);
diag(D)
L;

%%

% 4 - 3.75 * 0.2667^2 - 3.73333 * 0.0191388^2 - 3.73214 * 0.0714286^2 - 3.73206 * 0.287179^2

(-1 - (0.1007 * 0.3028 * 3.3022+ 0.0365 * 0.1005 *3.3338 + 0.2974 * 0.0099 * 3.7))/3.3362

-(0.0365 * 0.3000 * 3.3338 + 0.2974 * 0.0034 * 3.7)/3.3362

%%
% init
u = zeros(m*n, 1);
% backward euler
f = F(m, n, h);
steps = 10;
for i = 1:steps
    br = u + th*f;
    u = A\br;
end
imagesc(reshape(u, [m, n]))

% boundary terms
function y = F(m, n, h)
    y = zeros(m*n, 1);
    y(1) = a(h) + b(h);
    y(n) = a(n*h) + d(h);
    y((m-1)*n+1) = b(m*h) + c(h);
    y(m*n) = c(n*h) + d(m*h);

    % first line
    for i = 2:n-1
        y(i) = a(i*h);
    end
    % internal lines
    for i = 2:m-1
        y((i-1)*n+1) = b(i*h);
        y(i*n) = d(i*h);
    end
    % last line
    for i = 2:n-1
        y((m-1)*n+i) = c(i*h);
    end
end

% boundary conditions
% top
function y = a(x)
    y = 3;
end
% left
function y = b(x)
    y = 3;
end
% bottom
function y = c(x)
    y = 10;
end
% right
function y = d(x)
    y = 10;
end
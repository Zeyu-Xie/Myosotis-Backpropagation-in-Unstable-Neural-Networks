% 数值微分求 Jacobi 矩阵

f = @(x) 1.145*[x(1)*cos(pi/9)-x(2)*sin(pi/9); x(1)*sin(pi/9)+x(2)*cos(pi/9)];

x = [1;1];
x_1 = [1+(1e-6);1];
x_2 = [1;1+(1e-6)];

J = [1,0;0,1];

for i = 1:100
    x = [x, f(x(:,end))];
    x_1 = [x_1, f(x_1(:,end))];
    x_2 = [x_2, f(x_2(:,end))];
    J = [J, [x_1(:,end)-x(:,end), x_2(:,end)-x(:,end)]/(1e-6)];
end

for i = 1:20
    disp(i-1);
    disp(J(:,2*i-1:2*i));
end

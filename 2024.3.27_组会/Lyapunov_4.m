% 数值微分求 Jacobi 矩阵，然后算 Lyapunov 指数

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

Lyapunov = [];

format long;

for i = 2:90
    J_tmp = J(:,2*i-1:2*i);
    J_tmp = log(abs(eig(J_tmp)))/(i-1);
    disp(J_tmp);
    Lyapunov = [Lyapunov, J_tmp(1)]
end

% 绘制 Lyapunov 指数随迭代次数的变化图表
iterations = 1:89;
plot(iterations, Lyapunov);
xlabel('迭代次数');
ylabel('Lyapunov 指数');
title('Lyapunov 指数随迭代次数的变化');
grid on;

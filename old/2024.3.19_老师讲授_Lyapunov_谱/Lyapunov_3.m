% 数值微分求 Jacobi 矩阵，然后算 Lyapunov 谱

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
    Lyapunov = [Lyapunov, max(abs(J_tmp(1)), abs(J_tmp(2)))];
end

plot(1:89, Lyapunov, 'o-', 'LineWidth', 2, "DisplayName", "Lyapunov Exponent");
xlabel('n');
ylabel('Lyapunov Exponent');
title("Lyapunov Exponent");
legend('show');
grid on;
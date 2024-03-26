% 画轨迹图

f = @(x) 1.145*[x(1)*cos(pi/9)-x(2)*sin(pi/9); x(1)*sin(pi/9)+x(2)*cos(pi/9)];

x_0 = [1;1];
x_0_ = [1.3;1];

x = x_0;  % 将初始向量作为第一个元素
x_ = x_0_;

for i = 1:100
    x = [x, f(x(:, end))];  % 使用最后一个向量进行计算，并添加到列表中
    x_ = [x_, f(x_(:, end))];
end

disp(x);
disp(x_);

hold on;
plot(x(1,:), x(2,:), 'o-', 'LineWidth', 2, "DisplayName", "X");
plot(x_(1,:), x_(2,:), 'o-', 'LineWidth', 2, "DisplayName", "X with Error");
xlabel('x1');
ylabel('x2');
title('Trajectory Plot');
legend('show');
grid on;
hold off;
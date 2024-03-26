% QR 法计算 Lyapunov 谱

f = @(x) 1.145*[x(1)*cos(pi/9)-x(2)*sin(pi/9); x(1)*sin(pi/9)+x(2)*cos(pi/9)];

x = [1;1];
x_1 = [1+(1e-6);1];
x_2 = [1;1+(1e-6)];
J = cell(0);
J = [J, [1,0;0,1]];
e = cell(0);
e = [e, [1,0;0,1]];

for i=1:100
    x = [x, f(x(:,end))];
    x_1 = [x_1, f(x_1(:,end))];
    x_2 = [x_2, f(x_2(:,end))];
    J = [J, [(x_1(:, end)-x(:, end))/(1e-6), (x_2(:, end)-x(:, end))/(1e-6)]];
    e = [e, [J{end-1}*e{end}]];
end

for i=1:101
    disp(J{i});
end

for i=2:101
    [Q, R] = qr(e{i});
    disp(log(R(1,1))/(i-1));
end
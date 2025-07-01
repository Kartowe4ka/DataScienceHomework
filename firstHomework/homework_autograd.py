import torch

# 2.1 - Простые вычисления с градиентами
# Создайте тензоры x, y, z с параметром requires_grad=True
x = torch.rand(1, requires_grad=True)
y = torch.rand(1, requires_grad=True)
z = torch.rand(1, requires_grad=True)
# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = (x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z).sum()
# Найдите градиенты по всем переменным
f.backward()
# Проверьте результат аналитически
analytic_x = 2 * x + 2 * y * z
analytic_y = 2 * y + 2 * x * z
analytic_z = 2 * z + 2 * x * y
print(f'Для x: {torch.allclose(x.grad, analytic_x)}')
print(f'Для y: {torch.allclose(y.grad, analytic_y)}')
print(f'Для z: {torch.allclose(z.grad, analytic_z)}')


# 2.2 - Градиент функции потерь
# Реализуйте функцию MSE (среднеквадратичной ошибки):
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
y_true = torch.tensor([2.0, 4.0, 6.0, 8.0], requires_grad=True)
# где y_pred = w * x + b (линейная функция)
y_pred = w * x + b
# MSE = (1/n) * Σ(y_pred - y_true)^2
MSE = ((y_pred - y_true) ** 2).mean()
# Найдите градиенты по w и b
MSE.backward()
print(w.grad)
print(b.grad)


# 2.3 - Цепное правило
x = torch.rand(1, requires_grad=True)
# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
f = torch.sin(x ** 2 + 1)
# Найдите градиент df/dx
# Проверьте результат с помощью torch.autograd.grad
autograd = torch.autograd.grad(outputs=f, inputs=x, retain_graph=True)[0]
f.backward()
print(torch.allclose(autograd, x.grad))


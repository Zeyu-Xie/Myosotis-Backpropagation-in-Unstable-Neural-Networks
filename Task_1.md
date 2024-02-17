# Task 1

1. Find Examples on Gradient Explosion
   1. Before Reshet
   2. After Reshet
2. Read 3 Essays
   1. [How Chaotic Are Recurrent Neural Networks](./essays/How_Chaotic_are_Recurrent_Neural_Networks.pdf)
   2. [On the Difficulty of Training Recurrent Neural Networks](./essays/On_The_Dificulty_of_Training_Recurrent_Neural_Networks.pdf)
   3. [Batch Normalization: Accelerating  Deep Network Training by Reducing Internal Covariate Shift](./essays/Batch_Normalization_Accelerating _Deep_Network_Training_by_Reducing_Internal_Covariate_Shift.pdf)
3. Nonintrusive Adjoint Shadowing Algorithm
$$
x_{n+1}=f_\gamma(x_n),\ x_n\in\mathbb{R}\quad\Phi:\mathbb{R}^m\to\mathbb{R}
\\
\overline\Phi:=\lim_{t\to\infin}\frac{1}{T}\sum_{n=1}^{T}\Phi(x_n)
\\
\frac{\part\overline\Phi}{\part\gamma}\stackrel{?}{=}\lim_{T\to\infin}\frac{1}{T}\sum_{n=1}^{T}(\nu\cdot\frac{\part\Phi}{\part x})(x_n)
\\
\stackrel{N.I.S}{\Rarr}\nu=\nu'+\sum_{i=1}^{\phi}e_ia_i
$$

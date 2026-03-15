本文假定你已经知道了亥姆霍兹方程

1. paraxial approximation
https://en.wikipedia.org/wiki/Paraxial_approximation

2. The Fresnel diffraction integral is an exact solution to the paraxial Helmholtz equation 
https://en.wikipedia.org/wiki/Fresnel_diffraction

$$
U(x,y,z) = \frac{-i}{\lambda z} \iint_{-\infty}^{\infty} U(x_0, y_0, 0) \exp\left\{ \frac{i\pi}{\lambda z} \left[ (x-x_0)^2 + (y-y_0)^2 \right] \right\} dx_0 dy_0
$$

**菲涅尔衍射积分就像是“一元二次方程的求根公式”，而 Gaussian beam 就像是“某个具体方程算出来的根 $x=2$”。它们都和亥姆霍兹方程有关，但一个是求解方法，另一个是具体的解。**
将“理想的高斯函数” $U(x_0, y_0, 0)$ 和给定的位置 z 代入菲涅尔衍射积分，计算的结果就是在 z 处高斯光束的分布。

3. Gaussian beam is a solution of the paraxial Helmholtz equation
https://en.wikipedia.org/wiki/Helmholtz_equation#Paraxial_approximation

- 定义 $z=0$ 平面（束腰所在位置），初始光场的振幅分布是一个理想的高斯函数 $U(x_0, y_0, 0)$
- 初始光场代入上面的菲涅尔衍射积分,  $r^2 = x^2 + y^2$
$$
U(r,z) = U_0 \frac{w_0}{w(z)} \exp\left( -\frac{r^2}{w(z)^2} \right) \exp\left( -i\frac{kr^2}{2R(z)} \right) \exp(i\psi(z))
$$
为了公式 $U(r,z)$ 整洁，我们又引入了新的定义, 具体定义参考 4. gaussian beam 
- $w(z)$  : Beam waist

- $R(z)$  : Wavefront curvature  

- $\psi(z)$ : Gouy phase

4. gaussian beam 
https://en.wikipedia.org/wiki/Gaussian_beam

---

[示例视频]
[正确性验证]
# simulation_core.py
import numpy as np
import matplotlib.pyplot as plt
import io

def run_simulation(wavelength, numerical_aperture, dose):
    """
    这里是仿真逻辑的入口。
    实际中，你可能在这里调用 C++ 的 .so 库或者 subprocess 运行 C++ exe
    """
    
    # 1. 模拟计算过程 (假设这里调用了 C++ 算出了一个矩阵)
    # 模拟生成一个光强分布图 (Gaussian beam profile example)
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    
    # 使用输入的参数影响结果
    sigma = wavelength / numerical_aperture
    intensity = dose * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    # 2. 将结果转换为图像
    plt.figure(figsize=(6, 6))
    plt.imshow(intensity, cmap='viridis', extent=[-2, 2, -2, 2])
    plt.colorbar(label='Intensity')
    plt.title(f'Litho Simulation (NA={numerical_aperture})')
    
    # 将图像保存到内存 Buffer 中，而不是文件
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close()

    # 3. 返回关键指标
    critical_dimension = float(sigma * 0.5) # 假设计算出的 CD 值
    
    return img_buf, critical_dimension
### 欢迎使用 Simulation-toykits

Simulation-toykits 是基于 MIT 协议的开源网站, 该网站托管了一些光学仿真工具包（玩具包?）。

#### 材料数据库
材料数据来源于 [refractiveindex.info](https://refractiveindex.info/), 该数据库根据 [CC0 1.0通用公共领域专用](https://creativecommons.org/publicdomain/zero/1.0/)协议免费开放。这意味着你可以自由使用、修改和分发其内容，毫无限制，即使是商业目的，也无需许可。  

#### 多层膜传输矩阵应用

#### 高斯光束
- Hermite gaussain beam
- Laguerre gaussain beam
- Flat top beam

---

#### 项目部署
- 代码下载
``` shell
# 下载网页源代码
git clone https://github.com/likooooo/simulation_toykits.git
# 下载资源文件
git submodule update --init --recursive
```
- 初始化运行环境
``` shell
python3 -m venv env
source ./env/bin/activate
pip install --no-cache-dir -r requirements.txt
```

- 在本地启动服务
``` shell
streamlit run ./app.py --server.port=8501 --server.address=0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
# 访问网页 http://localhost:8501/
```

- 使用 docker 启动服务  
假定您已经安装了 docker 并配置好了镜像。
``` shell
docker build -t simulation-toykits:v1 .
docker run -p 8501:8501 simulation-toykits:v1
```
#### 其他配置


``` shell

# 开启 WSL 的 镜像网络模式，这样 WSL 和 Windows 将共享同一个 IP，局域网内就能直接访问了
[wsl2]
networkingMode=mirrored

# 即使 IP 一致了，Windows 的防火墙仍会拦截从外部进入 8501 端口的流量。你需要明确告诉 Windows 允许这个端口通过。
New-NetFirewallRule -DisplayName "Streamlit-Access" -Direction Inbound -LocalPort 8501 -Protocol TCP -Action Allow
```
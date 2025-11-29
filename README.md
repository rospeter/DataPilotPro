

# DataPilotPro

一个用于数据处理与分析的 Python 项目，基于 `customtkinter` GUI 和 `scikit-learn`，支持简单的数据可视化和回归测试。

---

## 🚀 项目功能

- GUI 数据输入与处理  
- 数据可视化（Matplotlib）  
- 简单的回归模型（Scikit-learn）  

---

## 📦 环境依赖

使用 Python 3.11+，依赖库如下：

```

customtkinter
matplotlib
numpy
pandas
scikit-learn

````

可通过 `requirements.txt` 一键安装全部依赖：

```bash
pip install -r requirements.txt
````

---

## 💻 小白使用指南（PyCharm）

### 1️⃣ 克隆项目

#### 方法 A：使用 Git

```bash
cd D:\Code\Python
git clone https://github.com/rospeter/DataPilotPro.git
```

#### 方法 B：下载 ZIP

1. 打开 GitHub 项目页面：[https://github.com/rospeter/DataPilotPro](https://github.com/rospeter/DataPilotPro)
2. 点击 **Code → Download ZIP**
3. 解压到你想存放的文件夹

---

### 2️⃣ 用 PyCharm 打开项目

1. 打开 **PyCharm**
2. 选择 **Open** → 选择 `DataPilotPro` 文件夹 → 点击 **OK**
3. 等待 PyCharm 加载项目

---

### 3️⃣ 配置虚拟环境（推荐）

1. 打开 **File → Settings → Project: DataPilotPro → Python Interpreter**
2. 点击右边齿轮 → **Add Interpreter → Virtualenv**
3. **Location** 默认即可
4. **Base interpreter** 选择你的 Python 3.11+
5. 点击 **Create**

---

### 4️⃣ 安装依赖

在 PyCharm **Terminal** 中执行：

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 5️⃣ 运行项目

#### 方法 A：在 PyCharm 中运行

1. 找到 `main.py` 文件
2. 右键 → **Run 'main'**

#### 方法 B：在 Terminal 中运行

```bash
python main.py
```

---

### 6️⃣ 常见问题

* **找不到 Python**

  * 确认 PyCharm 中 Python Interpreter 设置正确

* **缺少依赖报错**

  * 执行：

    ```bash
    pip install -r requirements.txt
    ```

* **GUI 不显示或黑屏**

  * 确认 `customtkinter` 已安装：

    ```bash
    pip install customtkinter
    ```

---

## 📝 项目结构

```
DataPilotPro/
├─ main.py
├─ requirements.txt
├─ random_test.csv
└─ .idea/
```

---


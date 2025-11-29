import customtkinter as ctk
from tkinter import messagebox, filedialog
import math, ast, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 全局外观配置 ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ========================
# AST 安全校验
# ========================
def validate_expression(expression):
    tree = ast.parse(expression, mode='eval')
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call,
        ast.Num, ast.Load, ast.Name, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
        ast.USub, ast.UAdd, ast.Mod
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"不允许的表达式结构: {type(node).__name__}")

# ========================
# 线性回归预测函数
# ========================
def linear_regression_predict(df, column_name, steps=5):
    """用 sklearn 线性回归预测未来 steps 个点"""
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[column_name].values
    model = LinearRegression()
    model.fit(X, y)
    X_future = np.arange(len(df), len(df) + steps).reshape(-1, 1)
    y_future = model.predict(X_future)
    return X.flatten(), y, X_future.flatten(), y_future

class DataPilotPro(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("DataPilot Pro - 商业级数据工具")
        self.geometry("1200x780")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.angle_mode = "rad"
        self.history = []

        self.create_sidebar()
        self.create_main_view()

    # =============================== Sidebar ===============================
    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        lbl_logo = ctk.CTkLabel(self.sidebar, text="DataPilot Pro", font=ctk.CTkFont(size=22, weight="bold"))
        lbl_logo.pack(pady=30)
        ctk.CTkButton(self.sidebar, text="首页概览", fg_color="transparent",
                      border_width=1, text_color=("gray10", "#DCE4EE")).pack(pady=10, padx=20, fill="x")
        ctk.CTkLabel(self.sidebar, text="Version 0.3", text_color="gray").pack(side="bottom", pady=20)

    # =============================== Tabs ===============================
    def create_main_view(self):
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.tab_basic = self.tabview.add("基础计算")
        self.tab_sci = self.tabview.add("科学计算")
        self.tab_data = self.tabview.add("数据可视化")
        self.tab_history = self.tabview.add("历史记录")
        self.tab_predict = self.tabview.add("线性回归预测")

        self.setup_basic_calculator_ui()
        self.setup_scientific_calculator_ui()
        self.setup_data_visualization_ui()
        self.setup_history_ui()
        self.setup_prediction_ui()

    # =============================== 基础计算器 ===============================
    def setup_basic_calculator_ui(self):
        self.basic_display = ctk.CTkEntry(
            self.tab_basic, width=400, height=70, font=("Arial", 32),
            justify="right", placeholder_text="0"
        )
        self.basic_display.pack(pady=30)
        self.basic_display.bind("<Return>", lambda event: self.calculate_result(self.basic_display))

        pad = ctk.CTkFrame(self.tab_basic, fg_color="transparent")
        pad.pack()

        buttons = [
            ('C', 0, 0, 'action'), ('(', 0, 1, 'op'), (')', 0, 2, 'op'), ('/', 0, 3, 'op'),
            ('7', 1, 0, 'num'), ('8', 1, 1, 'num'), ('9', 1, 2, 'num'), ('*', 1, 3, 'op'),
            ('4', 2, 0, 'num'), ('5', 2, 1, 'num'), ('6', 2, 2, 'num'), ('-', 2, 3, 'op'),
            ('1', 3, 0, 'num'), ('2', 3, 1, 'num'), ('3', 3, 2, 'num'), ('+', 3, 3, 'op'),
            ('DEL', 4, 0, 'action'), ('0', 4, 1, 'num'), ('.', 4, 2, 'num'), ('=', 4, 3, 'action'),
        ]
        self.create_keypad(pad, buttons, self.basic_display)

    # =============================== 科学计算器 ===============================
    def setup_scientific_calculator_ui(self):
        self.sci_display = ctk.CTkEntry(
            self.tab_sci, width=520, height=70,
            font=("Arial", 30), justify="right", placeholder_text="0"
        )
        self.sci_display.pack(pady=20)
        self.sci_display.bind("<Return>", lambda event: self.calculate_result(self.sci_display))

        sci_func_frame = ctk.CTkFrame(self.tab_sci, fg_color="transparent")
        sci_func_frame.pack(pady=10)

        sci_funcs = [
            ("sin", "sin("), ("cos", "cos("), ("tan", "tan("),
            ("log", "log("), ("ln", "log("),
            ("sqrt", "sqrt("), ("π", "3.141592653589793"),
            ("e", "2.718281828459045"), ("^", "**"),
        ]
        for i, (label, value) in enumerate(sci_funcs):
            btn = ctk.CTkButton(
                sci_func_frame, text=label,
                width=80, height=50, font=("Arial", 18, "bold"),
                fg_color="#5A4E9E", hover_color="#463C7D",
                command=lambda v=value: self.sci_display.insert("insert", v)
            )
            btn.grid(row=0, column=i, padx=5)

        base_pad = ctk.CTkFrame(self.tab_sci, fg_color="transparent")
        base_pad.pack(pady=10)
        base_buttons = [
            ('C', 0, 0, 'action'), ('(', 0, 1, 'op'), (')', 0, 2, 'op'), ('/', 0, 3, 'op'),
            ('7', 1, 0, 'num'), ('8', 1, 1, 'num'), ('9', 1, 2, 'num'), ('*', 1, 3, 'op'),
            ('4', 2, 0, 'num'), ('5', 2, 1, 'num'), ('6', 2, 2, 'num'), ('-', 2, 3, 'op'),
            ('1', 3, 0, 'num'), ('2', 3, 1, 'num'), ('3', 3, 2, 'num'), ('+', 3, 3, 'op'),
            ('DEL', 4, 0, 'action'), ('0', 4, 1, 'num'), ('.', 4, 2, 'num'), ('=', 4, 3, 'action'),
        ]
        self.create_keypad(base_pad, base_buttons, self.sci_display)

    # =============================== 历史记录 ===============================
    def setup_history_ui(self):
        self.history_box = ctk.CTkTextbox(self.tab_history, width=500, height=650)
        self.history_box.pack(pady=20, padx=20)

    # =============================== 数据可视化 ===============================
    def setup_data_visualization_ui(self):
        self.data_text = ctk.CTkTextbox(self.tab_data, width=600, height=140)
        self.data_text.pack(pady=10)

        btn_frame = ctk.CTkFrame(self.tab_data)
        btn_frame.pack(pady=5)

        ctk.CTkButton(btn_frame, text="导入 CSV", width=120,
                      command=self.load_csv).grid(row=0, column=0, padx=5)
        ctk.CTkButton(btn_frame, text="生成图表", width=120,
                      command=self.generate_chart).grid(row=0, column=1, padx=5)
        ctk.CTkButton(btn_frame, text="导出图片", width=120,
                      command=self.export_chart).grid(row=0, column=2, padx=5)

        self.chart_type = ctk.CTkOptionMenu(
            self.tab_data, values=["折线图", "柱状图", "饼图"]
        )
        self.chart_type.pack(pady=5)

        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_data)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(pady=10)

    # =============================== 线性回归预测 Tab ===============================
    def setup_prediction_ui(self):
        self.predict_data_text = ctk.CTkTextbox(self.tab_predict, width=600, height=140)
        self.predict_data_text.pack(pady=10)

        input_frame = ctk.CTkFrame(self.tab_predict)
        input_frame.pack(pady=5)

        ctk.CTkLabel(input_frame, text="预测列名:").grid(row=0, column=0, padx=5)
        self.predict_column_entry = ctk.CTkEntry(input_frame, width=120)
        self.predict_column_entry.grid(row=0, column=1, padx=5)

        ctk.CTkLabel(input_frame, text="预测步数:").grid(row=0, column=2, padx=5)
        self.predict_steps_entry2 = ctk.CTkEntry(input_frame, width=80)
        self.predict_steps_entry2.grid(row=0, column=3, padx=5)

        btn_frame = ctk.CTkFrame(self.tab_predict)
        btn_frame.pack(pady=5)

        ctk.CTkButton(btn_frame, text="导入 CSV", width=120, command=self.load_predict_csv).grid(row=0, column=0, padx=5)
        ctk.CTkButton(btn_frame, text="生成预测图", width=120, command=self.generate_prediction_chart).grid(row=0, column=1, padx=5)

        self.fig_pred, self.ax_pred = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas_pred = FigureCanvasTkAgg(self.fig_pred, master=self.tab_predict)
        self.canvas_pred.get_tk_widget().pack(pady=10)

    # =============================== CSV 载入 ===============================
    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV 文件", "*.csv")])
        if not path:
            return
        df = pd.read_csv(path)
        self.data_text.delete("1.0", "end")
        self.data_text.insert("end", df.to_csv(index=False))

    def load_predict_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV 文件", "*.csv")])
        if not path:
            return
        df = pd.read_csv(path)
        self.predict_data_text.delete("1.0", "end")
        self.predict_data_text.insert("end", df.to_csv(index=False))

    # =============================== 图表生成 ===============================
    def generate_chart(self):
        raw = self.data_text.get("1.0", "end").strip()
        if not raw:
            return
        from io import StringIO
        df = pd.read_csv(StringIO(raw))

        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        chart = self.chart_type.get()

        if chart == "折线图":
            for col in df.columns[1:]:
                self.ax.plot(df[df.columns[0]], df[col], marker='o', label=col)
            self.ax.set_title("折线图")
            self.ax.set_xlabel(df.columns[0])
            self.ax.set_ylabel("值")
            self.ax.legend()
        elif chart == "柱状图":
            df.plot(x=df.columns[0], kind="bar", ax=self.ax)
            self.ax.set_title("柱状图")
        elif chart == "饼图":
            for i, col in enumerate(df.columns[1:]):
                self.fig.clf()
                self.ax = self.fig.add_subplot(111)
                self.ax.pie(df[col], labels=df[df.columns[0]], autopct='%1.1f%%')
                self.ax.set_title(f"饼图 - {col}")
                self.canvas.draw()
                return
        self.canvas.draw()

    def generate_prediction_chart(self):
        raw = self.predict_data_text.get("1.0", "end").strip()
        if not raw:
            return
        from io import StringIO
        df = pd.read_csv(StringIO(raw))

        col_name = self.predict_column_entry.get()
        if col_name not in df.columns:
            messagebox.showerror("错误", f"列名 {col_name} 不存在")
            return

        steps = int(self.predict_steps_entry2.get() or 5)
        X, y, X_future, y_future = linear_regression_predict(df, col_name, steps)

        self.fig_pred.clf()
        self.ax_pred = self.fig_pred.add_subplot(111)
        self.ax_pred.plot(X, y, marker='o', label="原始数据")
        self.ax_pred.plot(X_future, y_future, linestyle="--", color="red", label="预测")
        self.ax_pred.set_title(f"{col_name} 线性回归预测")
        self.ax_pred.set_xlabel("索引")
        self.ax_pred.set_ylabel("值")
        self.ax_pred.legend()
        self.canvas_pred.draw()

    # =============================== 导出图像 ===============================
    def export_chart(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png")])
        if not path:
            return
        self.fig.savefig(path)

    # =============================== 通用按键生成 ===============================
    def create_keypad(self, parent, buttons, display):
        for (text, row, col, type_) in buttons:
            if type_ == 'num':
                fg, hov = "#3B8ED0", "#36719F"
            elif type_ == 'op':
                fg, hov = "#1F6AA5", "#144870"
            else:
                fg, hov = ("#2CC985", "#229965") if text == '=' else ("#E59400", "#B37400")

            btn = ctk.CTkButton(
                parent, text=text, width=80, height=60, font=("Arial", 20, "bold"),
                fg_color=fg, hover_color=hov,
                command=lambda t=text, d=display: self.on_btn_click(t, d)
            )
            btn.grid(row=row, column=col, padx=5, pady=5)

    # =============================== 按键逻辑 ===============================
    def on_btn_click(self, char, display):
        pos = display.index("insert")
        if char == 'C':
            display.delete(0, 'end')
            return
        if char == 'DEL':
            if pos > 0:
                display.delete(pos - 1)
            return
        if char == '=':
            self.calculate_result(display)
            return
        if char in ['sin', 'cos', 'tan', 'log', 'sqrt', 'abs', 'factorial', 'logb']:
            display.insert(pos, f"{char}(")
            return
        if char == 'π':
            display.insert(pos, "3.141592653589793")
            return
        if char == 'e':
            display.insert(pos, "2.718281828459045")
            return
        if char == '^':
            display.insert(pos, "**")
            return
        if char == 'ln':
            display.insert(pos, "log(")
            return
        display.insert(pos, char)

    # =============================== 计算逻辑（含智能括号补全） ===============================
    def calculate_result(self, display):
        expression = display.get()
        if not expression:
            return
        def auto_fix_parentheses(expr):
            left = expr.count("(")
            right = expr.count(")")
            if left > right:
                expr += ")" * (left - right)
            return expr
        expression = auto_fix_parentheses(expression)
        expression = expression.replace("^", "**")
        try:
            validate_expression(expression)
            safe_dict = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            safe_dict['abs'] = abs
            safe_dict['logb'] = lambda x, b=10: math.log(x, b)
            safe_dict['factorial'] = math.factorial

            if self.angle_mode == "deg":
                safe_dict['sin'] = lambda x: math.sin(math.radians(x))
                safe_dict['cos'] = lambda x: math.cos(math.radians(x))
                safe_dict['tan'] = lambda x: math.tan(math.radians(x))

            result = eval(expression, {"__builtins__": None}, safe_dict)
            if isinstance(result, float):
                result = round(result, 8)
                if result.is_integer():
                    result = int(result)

            display.delete(0, 'end')
            display.insert(0, str(result))
            self.history.append(f"{expression} = {result}")
            self.history_box.insert("end", f"{expression} = {result}\n")
            self.history_box.see("end")

        except Exception as e:
            messagebox.showerror("错误", f"计算表达式无效: {str(e)}")

    def on_closing(self):
        self.quit()
        self.destroy()

if __name__ == "__main__":
    app = DataPilotPro()
    app.mainloop()

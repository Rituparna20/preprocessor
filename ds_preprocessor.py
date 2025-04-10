import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import os, sys
from PIL import Image, ImageTk
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
import numpy as np

tk.sys = sys  # Fixes freezing issue on some systems

class DataPreprocessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Datanize")
        self.root.geometry("1000x800")
        self.root.configure(bg='#1e3d59')

        style = ttk.Style()
        style.theme_use('clam')

        try:
            icon_path = self.resource_path("logo.ico")
            self.root.iconbitmap(icon_path)
        except Exception as e:
            print(f"Icon load error: {e}")

        self.file_path = tk.StringVar()
        self.export_path = tk.StringVar()
        self.missing_values = {}
        self.scaling_columns = []
        self.feature_selection_columns = []
        self.logs = []

        self.feature_method = tk.StringVar(value="PCA")

        # Create a container frame for all left-aligned widgets
        main_frame = tk.Frame(root, bg='#1e3d59')
        main_frame.pack(anchor='w', padx=20, pady=10)

        tk.Label(main_frame, text="Select CSV File:", bg='#1e3d59', fg='white').grid(row=0, column=0, sticky='w', pady=5)
        file_frame = tk.Frame(main_frame, bg='#1e3d59')
        file_frame.grid(row=1, column=0, columnspan=2, sticky='w')
        tk.Entry(file_frame, textvariable=self.file_path, width=60).pack(side='left', pady=2)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side='left', padx=5)

        tk.Label(main_frame, text="Export Folder:", bg='#1e3d59', fg='white').grid(row=2, column=0, sticky='w', pady=5)
        export_frame = tk.Frame(main_frame, bg='#1e3d59')
        export_frame.grid(row=3, column=0, columnspan=2, sticky='w')
        tk.Entry(export_frame, textvariable=self.export_path, width=60).pack(side='left', pady=2)
        ttk.Button(export_frame, text="Browse", command=self.browse_folder).pack(side='left', padx=5)

        tk.Label(main_frame, text="Missing Value Strategy (Per Column):", bg='#1e3d59', fg='white').grid(row=4, column=0, sticky='w', pady=5)
        self.missing_frame = tk.Frame(main_frame, bg='#1e3d59')
        self.missing_frame.grid(row=5, column=0, columnspan=3, sticky='w')

        # Grid layout configuration to align dropdowns for missing values
        for i in range(5):
            self.missing_frame.grid_columnconfigure(i, weight=1)

        # Will populate this dynamically after file load

        tk.Label(main_frame, text="Feature Selection Columns:", bg='#1e3d59', fg='white').grid(row=6, column=0, sticky='w', pady=5)
        fs_frame = tk.Frame(main_frame, bg='#1e3d59')
        fs_frame.grid(row=7, column=0, columnspan=3, sticky='w')
        self.feature_selection_combo = ttk.Combobox(fs_frame, width=60, postcommand=self.update_columns_list)
        self.feature_selection_combo.pack(side='left', pady=2)
        ttk.Button(fs_frame, text="Add", command=self.add_feature_selection_column).pack(side='left', padx=5)
        ttk.Button(fs_frame, text="Undo", command=self.undo_feature_column).pack(side='left', padx=5)

        tk.Label(main_frame, text="Columns to Scale:", bg='#1e3d59', fg='white').grid(row=8, column=0, sticky='w', pady=5)
        scale_frame = tk.Frame(main_frame, bg='#1e3d59')
        scale_frame.grid(row=9, column=0, columnspan=3, sticky='w')
        self.scaling_combo = ttk.Combobox(scale_frame, width=60, postcommand=self.update_columns_list)
        self.scaling_combo.pack(side='left', pady=2)
        ttk.Button(scale_frame, text="Add", command=self.add_scaling_column).pack(side='left', padx=5)
        ttk.Button(scale_frame, text="Undo", command=self.undo_scaling_column).pack(side='left', padx=5)

        self.selected_features_label = tk.Label(main_frame, text="Selected Features: None", bg='#1e3d59', fg='white')
        self.selected_features_label.grid(row=10, column=0, sticky='w', pady=2)

        self.selected_scaling_label = tk.Label(main_frame, text="Selected for Scaling: None", bg='#1e3d59', fg='white')
        self.selected_scaling_label.grid(row=11, column=0, sticky='w', pady=2)

        tk.Label(main_frame, text="Feature Selection Method:", bg='#1e3d59', fg='white').grid(row=12, column=0, sticky='w', pady=5)
        self.method_combo = ttk.Combobox(main_frame, textvariable=self.feature_method, values=["PCA", "PLS", "Correlation"], width=20, state="readonly")
        self.method_combo.grid(row=13, column=0, sticky='w')
        self.method_combo.current(0)

        tk.Label(main_frame, text="Log Window", bg='#1e3d59', fg='white').grid(row=14, column=0, sticky='w', pady=5)
        self.log_box = tk.Text(main_frame, height=8, width=100, state='disabled', bg='black', fg='white', wrap='word')
        self.log_box.grid(row=15, column=0, columnspan=3, pady=5, sticky='w')

        ttk.Button(main_frame, text="Preprocess", command=self.preprocess).grid(row=16, column=0, pady=10, sticky='w')

        tk.Label(main_frame, text="Preview Window", bg='#1e3d59', fg='white').grid(row=17, column=0, sticky='w')
        self.preview = tk.Text(main_frame, height=10, width=120, state='disabled', bg='white')
        self.preview.grid(row=18, column=0, columnspan=3, pady=10, sticky='w')

        try:
            logo_path = self.resource_path("image.png")
            logo = Image.open(logo_path)
            logo = logo.resize((100, 100))
            self.logo_img = ImageTk.PhotoImage(logo)
            tk.Label(root, image=self.logo_img, bg='#1e3d59').place(relx=1.0, rely=0.0, anchor='ne')
        except Exception as e:
            print(f"Logo load error: {e}")

    def log(self, message):
        self.logs.append(message)
        self.log_box.configure(state='normal')
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.configure(state='disabled')
        self.log_box.see(tk.END)

    def resource_path(self, relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.file_path.set(path)
            self.df = pd.read_csv(path)
            self.populate_missing_strategy()

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.export_path.set(folder)

    def populate_missing_strategy(self):
        for widget in self.missing_frame.winfo_children():
            widget.destroy()
        self.strategy_options = {}
        for idx, col in enumerate(self.df.columns):
            if self.df[col].isnull().sum() > 0:
                r = idx // 4
                c = (idx % 4) * 2
                label_text = f"{col} (Missing: {self.df[col].isnull().sum()})"
                tk.Label(self.missing_frame, text=label_text, bg='#1e3d59', fg='white').grid(row=r, column=c, sticky='e')
                strategy = ttk.Combobox(self.missing_frame, values=["mean", "median", "most_frequent"], width=15)
                strategy.grid(row=r, column=c+1, padx=5, pady=2)
                self.strategy_options[col] = strategy

    def update_columns_list(self):
        if hasattr(self, 'df'):
            col_list = list(self.df.columns)
            self.feature_selection_combo['values'] = col_list
            self.scaling_combo['values'] = col_list

    def add_feature_selection_column(self):
        col = self.feature_selection_combo.get()
        if col and col not in self.feature_selection_columns:
            self.feature_selection_columns.append(col)
            self.selected_features_label.config(text=f"Selected Features: {', '.join(self.feature_selection_columns)}")

    def add_scaling_column(self):
        col = self.scaling_combo.get()
        if col and col not in self.scaling_columns:
            self.scaling_columns.append(col)
            self.selected_scaling_label.config(text=f"Selected for Scaling: {', '.join(self.scaling_columns)}")

    def undo_feature_column(self):
        if self.feature_selection_columns:
            removed = self.feature_selection_columns.pop()
            self.selected_features_label.config(text=f"Selected Features: {', '.join(self.feature_selection_columns)}")
            self.log(f"Removed feature selection column: {removed}")

    def undo_scaling_column(self):
        if self.scaling_columns:
            removed = self.scaling_columns.pop()
            self.selected_scaling_label.config(text=f"Selected for Scaling: {', '.join(self.scaling_columns)}")
            self.log(f"Removed scaling column: {removed}")

    def preprocess(self):
        try:
            df = self.df.copy()
            for col, combo in self.strategy_options.items():
                strategy = combo.get()
                if strategy:
                    imputer = SimpleImputer(strategy=strategy)
                    df[[col]] = imputer.fit_transform(df[[col]])
                    self.log(f"Applied {strategy} imputation on {col}")

            label_enc = LabelEncoder()
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = label_enc.fit_transform(df[col])
                self.log(f"Applied Label Encoding on {col}")

            if self.scaling_columns:
                scaler = StandardScaler()
                df[self.scaling_columns] = scaler.fit_transform(df[self.scaling_columns])
                self.log(f"Scaled columns: {', '.join(self.scaling_columns)}")

            if self.feature_selection_columns:
                if self.feature_method.get() == "PCA" and len(self.feature_selection_columns) >= 2:
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(df[self.feature_selection_columns])
                    df['PC1'], df['PC2'] = components[:, 0], components[:, 1]
                    self.log("Performed PCA")
                elif self.feature_method.get() == "PLS" and len(self.feature_selection_columns) >= 2:
                    pls = PLSRegression(n_components=2)
                    df_temp = df[self.feature_selection_columns].copy()
                    pls.fit(df_temp, np.random.rand(df_temp.shape[0]))
                    components = pls.transform(df_temp)
                    df['PLS1'], df['PLS2'] = components[:, 0], components[:, 1]
                    self.log("Performed PLS")
                elif self.feature_method.get() == "Correlation":
                    corr = df[self.feature_selection_columns].corr()
                    self.log(f"Correlation Matrix:\n{corr.to_string()}\n")

            y = df.iloc[:, -1]
            X = df.drop(columns=[df.columns[-1]])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train.to_csv(os.path.join(self.export_path.get(), "x-train.csv"), index=False)
            X_test.to_csv(os.path.join(self.export_path.get(), "x-test.csv"), index=False)
            y_train.to_csv(os.path.join(self.export_path.get(), "y-train.csv"), index=False)
            y_test.to_csv(os.path.join(self.export_path.get(), "y-test.csv"), index=False)

            self.preview.configure(state='normal')
            self.preview.delete("1.0", tk.END)
            self.preview.insert(tk.END, df.head().to_string(index=False))
            self.preview.configure(state='disabled')

            self.log("Files saved as x-train.csv, x-test.csv, y-train.csv, y-test.csv")
            messagebox.showinfo("Success", "Preprocessing completed and files saved.")
        except Exception as e:
            self.log(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = DataPreprocessorApp(root)
    root.mainloop()

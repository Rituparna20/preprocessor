import os
import pandas as pd
from tkinter import Tk, Label, Button, Entry, filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class PreprocessApp:
    def __init__(self, master):
        self.master = master
        master.title("Data Preprocessor")

        # CSV File Selection
        self.label_csv = Label(master, text="Select CSV File:")
        self.label_csv.grid(row=0, column=0, padx=10, pady=10)

        self.entry_csv = Entry(master, width=50)
        self.entry_csv.grid(row=0, column=1)

        self.btn_browse_csv = Button(master, text="Browse", command=self.browse_csv)
        self.btn_browse_csv.grid(row=0, column=2, padx=5)

        # Export Directory Selection
        self.label_export = Label(master, text="Select Export Folder:")
        self.label_export.grid(row=1, column=0, padx=10, pady=10)

        self.entry_export = Entry(master, width=50)
        self.entry_export.grid(row=1, column=1)

        self.btn_browse_export = Button(master, text="Browse", command=self.browse_export)
        self.btn_browse_export.grid(row=1, column=2, padx=5)

        # Process Button
        self.btn_process = Button(master, text="Preprocess", command=self.process)
        self.btn_process.grid(row=2, column=1, pady=20)

    def browse_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            self.entry_csv.delete(0, 'end')
            self.entry_csv.insert(0, filepath)

    def browse_export(self):
        folderpath = filedialog.askdirectory()
        if folderpath:
            self.entry_export.delete(0, 'end')
            self.entry_export.insert(0, folderpath)

    def process(self):
        csv_path = self.entry_csv.get()
        export_dir = self.entry_export.get()

        if not os.path.isfile(csv_path):
            messagebox.showerror("Error", "Invalid CSV file path.")
            return
        if not os.path.isdir(export_dir):
            messagebox.showerror("Error", "Invalid export directory.")
            return

        try:
            self.preprocess_csv(csv_path, export_dir)
            messagebox.showinfo("Success", "✅ Preprocessing completed successfully!")
        except Exception as e:
            messagebox.showerror("Exception", str(e))

    def preprocess_csv(self, csv_path, export_dir):
        df = pd.read_csv(csv_path)

        # Fill missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

        # Encode categorical columns
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        # Feature Scaling
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Export Processed Files
        os.makedirs(export_dir, exist_ok=True)
        pd.DataFrame(X_train).to_csv(os.path.join(export_dir, "X_train.csv"), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(export_dir, "X_test.csv"), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(export_dir, "y_train.csv"), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(export_dir, "y_test.csv"), index=False)


if __name__ == "__main__":
    root = Tk()
    app = PreprocessApp(root)
    root.geometry("700x180")
    root.mainloop()

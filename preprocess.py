import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from tkinter import Tk, filedialog, messagebox

def preprocess_csv(csv_path, export_dir):
    # Load CSV
    df = pd.read_csv(csv_path)
    print("Initial Data Info:")
    print(df.info())
    print("\nMissing Values:\n", df.isnull().sum())

    # Step 1: Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Step 2: Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Step 3: Feature Scaling
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 4: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Step 5: Export Processed Files
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    pd.DataFrame(X_train).to_csv(os.path.join(export_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(export_dir, "X_test.csv"), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(export_dir, "y_train.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(export_dir, "y_test.csv"), index=False)

    print("\n✅ Preprocessing Complete. Files saved to:", export_dir)

# -------------------------
# MAIN EXECUTION BLOCK
# -------------------------
if __name__ == "__main__":
    root = Tk()
    root.withdraw()  # Hide the main window

    try:
        # Ask user to select the CSV file
        csv_path = filedialog.askopenfilename(
            title="Select your CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not csv_path:
            messagebox.showerror("Error", "No CSV file selected.")
            raise SystemExit()

        # Ask user to choose export directory
        export_dir = filedialog.askdirectory(title="Select folder to save preprocessed files")

        if not export_dir:
            messagebox.showerror("Error", "No export folder selected.")
            raise SystemExit()

        # Proceed with processing
        preprocess_csv(csv_path, export_dir)
        messagebox.showinfo("Success", "✅ Preprocessing complete!")

    except Exception as e:
        messagebox.showerror("Unhandled Exception", str(e))

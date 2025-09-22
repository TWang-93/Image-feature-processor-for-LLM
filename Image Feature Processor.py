import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from torchvision import models
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


class ImageFeatureProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Feature Processor")
        self.root.geometry("1000x500")


        try:
            self.logo_img = Image.open("logo.png")
            self.logo_img = self.logo_img.resize((300, 100), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(self.logo_img)
        except:
            self.logo_photo = None


        self.train_dir = tk.StringVar()
        self.val_dir = tk.StringVar()
        self.test_dir = tk.StringVar()
        self.output_dir = tk.StringVar(value="./split_data")
        self.pca_components = tk.IntVar(value=10)  
        self.progress = tk.DoubleVar()

        self.create_widgets()
        self.load_model()

    def load_model(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.resnet = models.resnet18(pretrained=True)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1]) 
        self.resnet.eval()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        if self.logo_photo:
            logo_frame = ttk.Frame(main_frame)
            logo_frame.pack(fill=tk.X, pady=5)
            ttk.Label(logo_frame, image=self.logo_photo).pack()


        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)


        display_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="10")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(control_frame, text="Training Set Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(control_frame, textvariable=self.train_dir, width=30).grid(row=0, column=1, pady=5)
        ttk.Button(control_frame, text="Browse...", command=lambda: self.browse_directory(self.train_dir)).grid(row=0, column=2, pady=5)

        ttk.Label(control_frame, text="Validation Set Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(control_frame, textvariable=self.val_dir, width=30).grid(row=1, column=1, pady=5)
        ttk.Button(control_frame, text="Browse...", command=lambda: self.browse_directory(self.val_dir)).grid(row=1, column=2, pady=5)

        ttk.Label(control_frame, text="Test Set Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(control_frame, textvariable=self.test_dir, width=30).grid(row=2, column=1, pady=5)
        ttk.Button(control_frame, text="Browse...", command=lambda: self.browse_directory(self.test_dir)).grid(row=2, column=2, pady=5)

        ttk.Label(control_frame, text="Output Directory:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(control_frame, textvariable=self.output_dir, width=30).grid(row=3, column=1, pady=5)
        ttk.Button(control_frame, text="Browse...", command=lambda: self.browse_directory(self.output_dir, is_folder=True)).grid(row=3, column=2, pady=5)

        ttk.Label(control_frame, text="PCA Components:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(control_frame, from_=1, to=512, textvariable=self.pca_components, width=10).grid(row=4, column=1, sticky=tk.W, pady=5)

        ttk.Button(control_frame, text="Extract Features (ResNet18)", command=self.extract_features).grid(row=5, column=0, columnspan=3, pady=10, sticky=tk.EW)
        ttk.Button(control_frame, text="Apply PCA Dimensionality Reduction", command=self.apply_pca).grid(row=6, column=0, columnspan=3, pady=10, sticky=tk.EW)

        ttk.Progressbar(control_frame, variable=self.progress, maximum=100).grid(row=7, column=0, columnspan=3, pady=10, sticky=tk.EW)

        self.result_text = tk.Text(display_frame, wrap=tk.WORD, height=20)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=5)

        self.redirect_output()

    def redirect_output(self):

        import sys
        class StdoutRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
            def write(self, message):
                self.text_widget.insert(tk.END, message)
                self.text_widget.see(tk.END)
                self.text_widget.update_idletasks()
        sys.stdout = StdoutRedirector(self.result_text)


    def browse_directory(self, directory_var, is_folder=False):
        if is_folder:
            directory = filedialog.askdirectory()
        else:
            directory = filedialog.askdirectory()
        if directory:
            directory_var.set(directory)

    def extract_features(self):
        if not self.train_dir.get():
            messagebox.showerror("Error", "Please select training set directory first!")
            return
        try:
            self.progress.set(0)
            print("Starting feature extraction with ResNet18...")
            os.makedirs(self.output_dir.get(), exist_ok=True)

            for dataset_dir in [self.train_dir.get(), self.val_dir.get(), self.test_dir.get()]:
                if dataset_dir:
                    dataset_name = os.path.basename(dataset_dir.rstrip("/\\"))
                    output_csv = os.path.join(self.output_dir.get(), f"{dataset_name}_features.csv")
                    print(f"Processing dataset: {dataset_name}")
                    features, labels, paths = self.process_directory(dataset_dir)
                    self.save_features(features, labels, paths, output_csv)

            print("Feature extraction completed successfully!")
            messagebox.showinfo("Success", "Feature extraction completed!")
        except Exception as e:
            messagebox.showerror("Error", f"Feature extraction failed: {str(e)}")
            print(f"Error: {str(e)}")

    def process_directory(self, directory):
        features, labels, paths = [], [], []
        total_files = sum([len(files) for _, _, files in os.walk(directory)])
        processed_files = 0
        for class_name in ["0", "1"]:
            class_dir = os.path.join(directory, class_name)
            if not os.path.exists(class_dir):
                continue
            for file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, file)
                try:
                    image = Image.open(img_path).convert("RGB")
                    image = self.transform(image).unsqueeze(0)
                    with torch.no_grad():
                        feat = self.resnet(image).squeeze().numpy()
                    features.append(feat)
                    labels.append(int(class_name))
                    paths.append(img_path)
                    processed_files += 1
                    self.progress.set(processed_files / total_files * 100)
                    self.root.update()
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
        return np.array(features), np.array(labels), paths

    def save_features(self, features, labels, paths, output_path):
        df = pd.DataFrame(features)
        df["label"] = labels
        df["path"] = paths
        df.to_csv(output_path, index=False)
        print(f"Features saved to: {output_path}")

    def apply_pca(self):
        try:
            print("Starting PCA dimensionality reduction...")
            for dataset_dir in [self.train_dir.get(), self.val_dir.get(), self.test_dir.get()]:
                if dataset_dir:
                    dataset_name = os.path.basename(dataset_dir.rstrip("/\\"))
                    features_csv = os.path.join(self.output_dir.get(), f"{dataset_name}_features.csv")
                    if not os.path.exists(features_csv):
                        continue
                    df = pd.read_csv(features_csv)
                    feature_cols = [c for c in df.columns if c not in ["label", "path"]]
                    features = df[feature_cols].values
                    pca = PCA(n_components=self.pca_components.get())
                    pca_features = pca.fit_transform(features)
                    self.save_pca_results(pca_features, df, dataset_name)
            print("PCA dimensionality reduction completed successfully!")
            messagebox.showinfo("Success", "PCA dimensionality reduction completed!")
        except Exception as e:
            messagebox.showerror("Error", f"PCA failed: {str(e)}")
            print(f"Error: {str(e)}")

    def save_pca_results(self, pca_features, original_df, dataset_name):
        pca_cols = [f"pca_{i+1}" for i in range(pca_features.shape[1])]
        pca_df = pd.DataFrame(pca_features, columns=pca_cols)
        pca_df["label"] = original_df["label"].values
        pca_df["path"] = original_df["path"].values
        output_path = os.path.join(
            self.output_dir.get(),
            f"{dataset_name}_resnet_pca{self.pca_components.get()}.csv"
        )
        pca_df.to_csv(output_path, index=False)
        print(f"PCA results saved to: {output_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageFeatureProcessorApp(root)
    root.mainloop()

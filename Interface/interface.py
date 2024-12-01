import csv
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkhtmlview import HTMLLabel, HTMLText, RenderHTML
import torch
import torch.nn.functional as F
from student_prediction_model import StudentPredictionModel

import pdb

import pandas as pd

class StudentPredictionApplication:
    def __init__(self, root, model):
        self.last_prediction = None
        self.fields = None
        self.input_frame = None
        self.btn_frame = None
        self.icon = None
        self.tos_frame = None
        self.root = root
        self.root.title("Student Success Predictor")
        self.model = model
        self.create_terms_of_service()
        self.icon = tk.PhotoImage(file="resources/Brock.png").subsample(6, 6)

    def create_terms_of_service(self):
        """
        Creates the Terms of Service window that must be agreed to
        """
        # Build the TOS window
        self.tos_frame = ttk.Frame(self.root)
        self.tos_frame.pack(expand=True, fill="both")

        # Load in the HTML version of TOS
        label = HTMLText(self.tos_frame, html=RenderHTML('resources/html/tos_html.html'))
        label.pack(expand=True, fill="both", padx=40, pady=40)

        # Agree and disagree buttons
        btn_frame = ttk.Frame(self.tos_frame, padding=4)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="I Disagree", command=self.close_app).pack(side=tk.LEFT, padx=(5, 10))
        ttk.Button(btn_frame, text="I Agree", style="Accent.TButton", command=self.open_main_app).pack(side=tk.RIGHT,
                                                                                                       padx=(5, 10))

    def open_main_app(self):
        """
        Closes Terms of Service frame and opens the main application.
        """
        self.tos_frame.destroy()
        self.create_main_ui()

    def close_app(self):
        """
        Closes the entire application.
        """
        self.root.destroy()

    def create_main_ui(self):
        """
        Creates the main application interface
        """
        # Menu bar at the top
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New", command=self.clear_inputs)
        file_menu.add_command(label="Save", command=self.save_to_csv)
        file_menu.add_command(label="Save & Exit", command=self.save_and_exit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        # Help Menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Terms of Service",
                              command=lambda: self.show_popup_from_file("Terms of Service", "resources/html/tos_html"
                                                                                            ".html"))
        help_menu.add_command(label="Privacy Notice",
                              command=lambda: self.show_popup_from_file("Privacy Notice", "resources/html/priv_html"
                                                                                          ".html"))
        help_menu.add_command(label="About this Software",
                              command=lambda: self.show_popup_from_file("About this Software", "resources/html"
                                                                                               "/info_html.html"))
        menu_bar.add_cascade(label="Help", menu=help_menu)

        icon_label = tk.Label(self.root, image=self.icon)
        icon_label.pack(anchor='w')

        # Main UI Components
        ttk.Label(self.root, text="Welcome to the Student Outcome Predictor!", font=("Arial", 16)).pack(pady=10)
        ttk.Label(self.root, text="Enter student information below manually, or click 'upload' to upload a students "
                                  "information from a CSV file. Once all the information fields are filled, "
                                  "click 'predict' to run the AI prediction tool.", font=("Arial", 12)).pack(pady=5)

        # Command button frame
        self.btn_frame = ttk.Frame(self.root, padding=10)
        self.btn_frame.pack(pady=10)

        # Upload Button
        ttk.Button(self.btn_frame, text="Upload", command=self.upload_csv).pack(side=tk.LEFT, padx=5, pady=10)

        # Prediction Button
        ttk.Button(self.btn_frame, text="Predict", style="Accent.TButton", command=self.predict).pack(side=tk.RIGHT,
                                                                                                      padx=5, pady=10)

        # Batch Predict Button
        ttk.Button(self.btn_frame, text="Batch Predict", style="Accent.TButton", command=self.batch_predict).pack(
            side=tk.RIGHT,
            padx=5, pady=10)

        # Input frame for text fields
        self.input_frame = ttk.Frame(self.root, padding=10)
        self.input_frame.pack(pady=10)

        # Input Fields for Student Information
        self.fields = []
        field_names = [
            "Marital status", "Application mode", "Application order", "Course", "Daytime/evening attendance",
            "Previous qualification", "Previous qualification (grade)", "Mother's qualification",
            "Father's qualification",
            "Mother's occupation", "Father's occupation", "Admission grade", "Displaced", "Educational special needs",
            "Debtor", "Tuition fees up to date", "Scholarship holder", "Age at enrollment", "International",
            "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (evaluations)",
            "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
            "Curricular units 1st sem (without evaluations)",
            "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
            "Curricular units 2nd sem (evaluations)",
            "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
            "Curricular units 2nd sem (without evaluations)",
            "Unemployment rate", "Inflation rate", "GDP"
        ]

        # Draw the input fields sequentially on the frame
        columns = 5
        for idx, field_name in enumerate(field_names):
            row = idx // columns
            column = idx % columns
            frame = ttk.Frame(self.input_frame, padding=2)
            frame.grid(row=row, column=column, sticky="s", padx=5, pady=10)
            ttk.Label(frame, text=f"{field_name}:", font=("Arial", 10)).pack(anchor="s")
            entry = ttk.Entry(frame, width=35, font=("Arial", 10))
            entry.pack(fill=tk.X, anchor="s", expand=True)
            entry.bind("<FocusIn>", lambda e, field=entry: self.clear_placeholder(field))
            self.fields.append(entry)

        # Privacy Notice Summary
        ttk.Label(self.root, text="Disclaimer: By using this software, you acknowledge and agree to our Terms of "
                                  "Service and Privacy Notice. We collect and use your personal data, "
                                  "including contact and usage information, to improve our services and ensure a "
                                  "secure user experience. Your data may be shared with service providers, "
                                  "affiliates, or partners as outlined in the Privacy Notice. We are committed to "
                                  "safeguarding your data with robust security measures and providing you with rights "
                                  "to access, correct, or delete your information. For more details, please refer to "
                                  "our full Privacy Notice and Terms of Service.",
                  font=("Arial", 10), foreground="gray", wraplength=1300).pack(side=tk.BOTTOM, pady=5)

    def clear_inputs(self):
        """
        Clears all input fields.
        """
        for field in self.fields:
            field.delete(0, tk.END)
            field.configure(foreground="white", background="white")

    def save_to_csv(self):
        """
        Saves input data and predictions to a CSV file.
        """
        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                # Retrieve all input data
                input_data = self.get_input_fields()
                input_values = [input_data[name] for name in input_data.keys()]

                # If prediction exists, append it
                if hasattr(self, 'last_prediction') and hasattr(self, 'last_confidence'):
                    input_values.append(self.last_prediction)
                    input_values.append(f"{self.last_confidence * 100:.2f}%")

                # Define headers for CSV
                headers = list(input_data.keys())
                if hasattr(self, 'last_prediction') and hasattr(self, 'last_confidence'):
                    headers += ["Predicted Outcome", "Confidence Score"]

                # Write to CSV
                with open(file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(headers)
                    writer.writerow(input_values)
                messagebox.showinfo("Save Successful", "Data saved successfully!")
            except Exception as e:
                messagebox.showerror("Save Error", f"An error occurred while saving: {str(e)}")

    def upload_csv(self):
        """
        Uploads a CSV file and populates the input fields with its data.
        """
        file_path = filedialog.askopenfilename(defaultextension=".csv",
                                               filetypes=[("CSV files", "*.csv")])
        # As long as the csv is not empty, cycle through, populate each field, alert for empty fields
        if file_path:
            with open(file_path, mode='r') as file:
                reader = csv.reader(file)
                data = list(reader)
                if len(data) > 1:
                    values = data[1]
                    for i, field in enumerate(self.fields):
                        if values[i].strip():
                            field.delete(0, tk.END)
                            field.insert(0, values[i])
                            field.configure(foreground="white", background="white")
                        else:
                            field.delete(0, tk.END)
                            field.insert(0, "*****")
                            field.configure(foreground="red")

    def clear_placeholder(self, field):
        """
        Clears the placeholder text when the user focuses on the entry field.
        """
        if field.get() == "*****":
            field.delete(0, tk.END)
            field.configure(foreground="white")

    def save_and_exit(self):
        """
        Saves data and closes the application.
        """
        self.save_to_csv()
        self.close_app()

    def show_popup_from_file(self, title, file_path):
        """
        Displays a pop-up window with content read from an HTML file.
        """
        # Create popup field
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("1200x600")
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)

        # Read the file being rendered
        with open(file_path, "r", encoding='UTF-8') as file:
            html_content = file.read()

        # Render the html file in the popup field
        label = HTMLLabel(popup, html=html_content)
        label.pack(expand=True, fill="both")
        ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=5)

    def get_input_fields(self):
        """
        Retrieves current values from all input fields.
        """
        input_data = {}
        field_names = [
            "Marital status", "Application mode", "Application order", "Course", "Daytime/evening attendance",
            "Previous qualification", "Previous qualification (grade)", "Mother's qualification",
            "Father's qualification",
            "Mother's occupation", "Father's occupation", "Admission grade", "Displaced", "Educational special needs",
            "Debtor", "Tuition fees up to date", "Scholarship holder", "Age at enrollment", "International",
            "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (evaluations)",
            "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
            "Curricular units 1st sem (without evaluations)",
            "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
            "Curricular units 2nd sem (evaluations)",
            "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
            "Curricular units 2nd sem (without evaluations)",
            "Unemployment rate", "Inflation rate", "GDP"
        ]

        for name, field in zip(field_names, self.fields):
            input_data[name] = field.get()

        return input_data

    def batch_predict(self):
        """
        Handles batch prediction logic by reading in multiple rows from a CSV file and saving the results.
        """
        file_path = filedialog.askopenfilename(defaultextension=".csv",
                                               filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                # Read CSV file with input data
                with open(file_path, mode='r') as file:
                    reader = csv.reader(file)
                    data = list(reader)
                    headers = data[0]
                    rows = data[1:]

                    # Prepare results list
                    results = []
                    results.append(headers + ["Predicted Outcome", "Confidence Score"])

                    # Loop through each row of data for prediction
                    for values in rows:
                        # Ensure all features are properly formatted and converted to floats
                        input_features = [float(value) if value.strip() else 0.0 for value in values]

                        # Apply L2 normalization
                        input_features = F.normalize(torch.tensor(input_features, dtype=torch.float32), p=2,
                                                     dim=0).tolist()

                        # Make a prediction
                        predicted_label, confidence_score = self.model.predict(input_features)

                        # Append the prediction and confidence to the row
                        values.append(predicted_label)
                        values.append(f"{confidence_score * 100:.2f}%")

                        # Append the complete row to results
                        results.append(values)

                # Save the results to a new CSV file
                save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                         filetypes=[("CSV files", "*.csv")])
                if save_path:
                    with open(save_path, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(results)
                    messagebox.showinfo("Batch Prediction Successful", "Batch predictions saved successfully!")
            except Exception as e:
                messagebox.showerror("Batch Prediction Error", f"An error occurred: {str(e)}")

    def predict(self):
        """
        Handles prediction logic.
        """
        input_data = self.get_input_fields()
        # Convert the input data dictionary to a list of values
        input_features = []
        for value in input_data.values():
            if value and value.strip():  # Check if the value is not None or only whitespace
                try:
                    input_features.append(float(value))
                except ValueError:
                    messagebox.showerror("Input Error", f"Invalid value in input field: {value}")
                    return
            else:
                messagebox.showerror("Input Error", "Not all text fields are filled.")
                return

        # Ensure the length matches the expected input size of the model
        if len(input_features) != 34:
            messagebox.showerror("Input Error", "The number of input features must be 34.")
            return

        try:
            predicted_label, confidence_score = self.model.predict(input_features)
            self.last_prediction = predicted_label  # Store prediction for saving
            self.last_confidence = confidence_score  # Store confidence score for saving
            self.display_results(predicted_label, confidence_score)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def display_results(self, prediction, confidence):
        """
        Displays the prediction results in a styled popup window.
        """
        # Create popup field
        popup = tk.Toplevel(self.root)
        popup.title("Prediction Results")
        popup.geometry("600x300")
        popup.configure(bg="#333333")  # Set background to match the application's dark theme
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)

        # Add a styled title for the results
        title_label = ttk.Label(popup, text="Prediction Results", font=("Arial", 18, "bold"), foreground="#FFFFFF",
                                background="#333333")
        title_label.pack(pady=15)

        # Render the prediction result and confidence score
        result_label = ttk.Label(popup, text=f"Predicted Outcome: {prediction}", font=("Arial", 14),
                                 foreground="#FFFFFF", background="#333333")
        result_label.pack(pady=10)
        confidence_label = ttk.Label(popup, text=f"Confidence Score: {confidence * 100:.2f}%", font=("Arial", 14),
                                     foreground="#FFFFFF", background="#333333")
        confidence_label.pack(pady=10)

        # Close Button
        close_button = ttk.Button(popup, text="Close", command=popup.destroy)
        close_button.pack(pady=10)


def main():
    # Load model for use in the backend
    input_mean = pd.read_csv('input_mean.csv')['0']
    input_std = pd.read_csv('input_std.csv')['0']
    model = StudentPredictionModel("resources/model.pth", input_mean=input_mean, input_std=input_std, logfile="confidence.csv")

    # Initialize the tkinter window
    root = tk.Tk()
    root.geometry('1632x918')
    root.tk.call("source", "resources/Azure-ttk-theme-main/azure.tcl")
    root.tk.call("set_theme", "dark")
    app = StudentPredictionApplication(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()

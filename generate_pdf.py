from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)  # Reduced font size
        self.cell(0, 10, 'Graph Attention Autoencoder for MultiOmics Integration,', 0, 1, 'C')  # First line of title
        self.cell(0, 10, 'Risk Stratification and Biomarker Identification in Cancer', 0, 1, 'C')  # Second line of title
        self.ln(5)

    def footer(self):
        self.set_y(-15)  # Position footer 15mm from the bottom
        self.set_font('Arial', 'I', 8)  # Italic font for footer
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')  # Centered page number

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)  # Regular font for readability
        body = body.replace('—', '-').replace('’', "'").replace('“', '"').replace('”', '"')
        self.multi_cell(0, 8, body)
        self.ln()

# File paths to include in the document
files_to_include = [
    "graph_autoencoder.py",
    "GraphAnalysis.py",
    "Attention_Extracter.py",
    "GATv2EncoderModel.py",
    "GATv2DecoderModel.py",
]

# Base directory of your project
base_dir = "/home/vpp1/Documents/Github/Graph-Attention-Autoencoder-for-MultiOmics-Analysis"

# Initialize PDF
pdf = PDF()
pdf.add_page()

# Process each file
for file_name in files_to_include:
    file_path = os.path.join(base_dir, file_name)
    if os.path.exists(file_path):
        pdf.chapter_title(f"File: {file_name}")
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            # Add first 75 lines
            pdf.chapter_body("First 75 lines:\n" + "".join(lines[:75]))
            
            # Add last 75 lines
            pdf.chapter_body("Last 75 lines:\n" + "".join(lines[-75:]))

# Save the PDF
output_path = os.path.join(base_dir, "GraphAttentionAutoencoder_Code.pdf")
pdf.output(output_path)
print(f"PDF generated: {output_path}")


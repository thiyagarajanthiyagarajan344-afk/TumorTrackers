
"""PDF report generator using ReportLab."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def _wrap_text(text: str, width: int):
    words = text.split()
    line = []
    count = 0
    for w in words:
        if count + len(w) + 1 > width:
            yield " ".join(line)
            line = [w]
            count = len(w)
        else:
            line.append(w)
            count += len(w) + 1
    if line:
        yield " ".join(line)

def build_pdf(pdf_path: str, patient_id, tumor_type: str, benign_malignant: str, confidence: float, summary: str, gradcam_path: str = None, created_at: str = None):
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    c.setTitle("Brain Tumor AI Report")
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 60, "Brain Tumor AI Analysis Report")
    c.setFont("Helvetica", 10)
    if created_at:
        c.drawString(40, height - 80, f"Generated: {created_at}")
    if patient_id:
        c.drawString(40, height - 95, f"Patient: {patient_id}")
    # Key box
    c.setStrokeColor(colors.black)
    c.setFillColor(colors.whitesmoke)
    c.rect(35, height - 220, width - 70, 110, fill=1)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 140, f"Tumor Type: {tumor_type}")
    c.drawString(50, height - 160, f"Benign/Malignant: {benign_malignant}")
    c.drawString(50, height - 180, f"Confidence: {confidence:.2%}")
    # Gradcam image
    if gradcam_path and os.path.exists(gradcam_path):
        try:
            img = ImageReader(gradcam_path)
            img_w, img_h = img.getSize()
            aspect = img_h / float(img_w)
            target_w = width - 70
            target_h = target_w * aspect
            c.drawImage(img, 35, height - 240 - target_h, width=target_w, height=target_h)
        except Exception:
            pass
    # Summary text
    text_obj = c.beginText(40, 190)
    text_obj.setFont("Helvetica", 11)
    for line in _wrap_text(summary, 90):
        text_obj.textLine(line)
    c.drawText(text_obj)
    c.showPage()
    c.save()
    return pdf_path

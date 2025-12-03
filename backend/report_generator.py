import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime

def create_bar_chart(data, title, ylabel, filename):
    """Creates a bar chart and saves it to a BytesIO buffer."""
    names = [item['Model'] for item in data]
    values = [item[ylabel] for item in data]
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, values, color='#0288d1')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    plt.close()
    buf.seek(0)
    return buf

def generate_report(insights_data):
    """Generates a PDF report from the insights data."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#0288d1')
    )
    story.append(Paragraph("Piezo.AI Discovery Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Introduction
    story.append(Paragraph("This report summarizes the performance of various machine learning models trained on your piezoelectric dataset. It includes comparisons for key properties like d33 and Tc.", styles['Normal']))
    story.append(Spacer(1, 24))

    # Process d33 Data
    if insights_data.get('comparison_d33'):
        story.append(Paragraph("d33 (Piezoelectric Coefficient) Analysis", styles['Heading2']))
        d33_results = insights_data['comparison_d33']
        
        # Table
        table_data = [['Model', 'R2 Score', 'RMSE']]
        for res in d33_results:
            table_data.append([res['Model'], f"{res['R2']:.4f}", f"{res['RMSE']:.4f}"])
        
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e1f5fe')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 24))

        # Charts
        img_buf_r2 = create_bar_chart(d33_results, 'd33 Model Comparison - R2 Score', 'R2', 'd33_r2.png')
        img_r2 = Image(img_buf_r2, width=5*inch, height=3.3*inch)
        story.append(img_r2)
        story.append(Spacer(1, 12))

    # Process Tc Data
    if insights_data.get('comparison_tc'):
        story.append(Paragraph("Tc (Curie Temperature) Analysis", styles['Heading2']))
        tc_results = insights_data['comparison_tc']
        
        # Table
        table_data = [['Model', 'R2 Score', 'RMSE']]
        for res in tc_results:
            table_data.append([res['Model'], f"{res['R2']:.4f}", f"{res['RMSE']:.4f}"])
        
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#fce4ec')), # Pinkish for Tc
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 24))

        # Charts
        img_buf_r2 = create_bar_chart(tc_results, 'Tc Model Comparison - R2 Score', 'R2', 'tc_r2.png')
        img_r2 = Image(img_buf_r2, width=5*inch, height=3.3*inch)
        story.append(img_r2)

    doc.build(story)
    buffer.seek(0)
    return buffer

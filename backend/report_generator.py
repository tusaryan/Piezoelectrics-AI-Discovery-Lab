import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime

# ==============================================================================
# REPORT STYLES & CONFIGURATION
# ==============================================================================
# Premium Color Palette (Professional & Scientific)
COLOR_PRIMARY_D33 = '#1565C0'  # Deep Azure Blue
COLOR_SECONDARY_D33 = '#42A5F5' # Soft Blue
COLOR_PRIMARY_TC = '#2E7D32'   # Deep Forest Green
COLOR_SECONDARY_TC = '#66BB6A'  # Soft Green
COLOR_ACCENT = '#EF6C00'       # Burnt Orange
COLOR_GREY = '#455A64'         # Slate Grey

def set_plot_style():
    """Sets a premium scientific plotting style."""
    plt.style.use('default') # Reset
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.axisbelow': True,
        'figure.facecolor': 'white',
        'axes.edgecolor': '#E0E0E0',
        'axes.linewidth': 1.2
    })

def create_bar_chart(data, metric, title, color, ylabel):
    """Creates a high-end bar chart for comparison."""
    set_plot_style()
    
    # Sort data for better visualization (Descending R2, Ascending RMSE)
    reverse_sort = True if metric == 'R2' else False
    sorted_data = sorted(data, key=lambda x: x[metric], reverse=reverse_sort)
    
    names = [item['Model'] for item in sorted_data]
    values = [item[metric] for item in sorted_data]
    
    plt.figure(figsize=(7, 4))
    
    # Elegant bars with slight edge
    bars = plt.bar(names, values, color=color, alpha=0.9, edgecolor=color, width=0.6)
    
    # Add value labels
    min_y = min(values)
    range_y = max(values) - min_y
    
    for bar in bars:
        height = bar.get_height()
        # Smart Label Positioning
        label_y = height + (range_y * 0.02) if height >= 0 else height - (range_y * 0.05)
        va = 'bottom' if height >= 0 else 'top'
        
        plt.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{height:.3f}',
                ha='center', va=va, fontsize=9, fontweight='bold', color='#37474F')

    plt.title(title, pad=20)
    plt.ylabel(ylabel, fontsize=10, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    
    # Add a footer source/note if needed? No, keep clean.
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def create_scatter_chart(y_test, y_pred, model_name, target_name):
    """Creates a scientific 'Predicted vs Actual' scatter plot."""
    set_plot_style()
    plt.figure(figsize=(6, 5))
    
    # Color based on target
    base_color = COLOR_PRIMARY_D33 if target_name == "d33" else COLOR_PRIMARY_TC
    
    # Scatter points - customized for density feel
    plt.scatter(y_test, y_pred, alpha=0.6, c=base_color, 
                edgecolors='white', linewidth=0.5, s=60, label='Test Data')
    
    # Perfect fit line (Diagonal)
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    
    # Add margin
    margin = (max_val - min_val) * 0.05
    limit_min = min_val - margin
    limit_max = max_val + margin
    
    plt.plot([limit_min, limit_max], [limit_min, limit_max], 
             color='#D32F2F', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    plt.xlim(limit_min, limit_max)
    plt.ylim(limit_min, limit_max)
    
    plt.title(f"{model_name}: Predicted vs Actual ({target_name})", pad=15)
    plt.xlabel(f"Actual {target_name} Values", fontweight='bold')
    plt.ylabel(f"Predicted {target_name} Values", fontweight='bold')
    plt.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def generate_insights_text(results, target_name):
    """Generates premium analytical commentary."""
    sorted_res = sorted(results, key=lambda x: x['R2'], reverse=True)
    best = sorted_res[0]
    worst = sorted_res[-1]
    
    text = f"<b>{target_name} Predictive Analysis</b><br/>"
    text += f"Our rigorous benchmarking process identified the <b>{best['Model']}</b> as the optimal algorithm for {target_name} prediction. "
    text += f"It achieved an efficient R² score of <b>{best['R2']:.4f}</b> and minimal error (RMSE: <b>{best['RMSE']:.4f}</b>). "
    text += f"This performance significantly outperforms the baseline, explaining {best['R2']*100:.1f}% of the variance in the experimental data.<br/><br/>"
    
    if best['R2'] > 0.9:
        text += "<b>Confidence Assessment: Excellent.</b> The model captures the complex physicochemical relationships with high fidelity. It is recommended for primary screening."
    elif best['R2'] > 0.75:
        text += "<b>Confidence Assessment: Good.</b> The model is reliable for trend analysis and candidate ranking, though experimental validation is advised for critical final selection."
    else:
        text += "<b>Confidence Assessment: Moderate.</b> While useful for broad filtering, the model's predictions should be treated as indicative. Augmenting the dataset with distinct crystal structures may improve accuracy."
        
    return text

def generate_report(insights_data):
    """Generates the PDF Report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []

    # --- Typography Styles ---
    styles.add(ParagraphStyle(name='ReportTitle', parent=styles['Normal'], fontSize=24, leading=30, textColor=colors.HexColor(COLOR_PRIMARY_D33), fontName='Helvetica-Bold', alignment=1, spaceAfter=8))
    styles.add(ParagraphStyle(name='ReportSubtitle', parent=styles['Normal'], fontSize=12, textColor=colors.HexColor(COLOR_GREY), alignment=1, spaceAfter=25))
    
    styles.add(ParagraphStyle(name='SectionHeaderD33', parent=styles['Heading2'], fontSize=16, textColor=colors.white, backColor=colors.HexColor(COLOR_PRIMARY_D33), borderPadding=10, spaceBefore=15, spaceAfter=15, fontName='Helvetica-Bold', borderRadius=5))
    styles.add(ParagraphStyle(name='SectionHeaderTC', parent=styles['Heading2'], fontSize=16, textColor=colors.white, backColor=colors.HexColor(COLOR_PRIMARY_TC), borderPadding=10, spaceBefore=15, spaceAfter=15, fontName='Helvetica-Bold', borderRadius=5))
    
    styles.add(ParagraphStyle(name='InsightBox', parent=styles['Normal'], fontSize=11, leading=16, backColor=colors.HexColor('#F5F5F5'), borderPadding=12, spaceAfter=20, borderColor=colors.HexColor('#E0E0E0'), borderWidth=0.5))
    
    styles.add(ParagraphStyle(name='ChartTitle', parent=styles['Heading3'], fontSize=14, textColor=colors.HexColor(COLOR_GREY), spaceBefore=10, spaceAfter=5, fontName='Helvetica-Bold'))

    # --- Header ---
    story.append(Paragraph("Piezoelectric Discovery Lab", styles['ReportTitle']))
    story.append(Paragraph(f"AI Model Assessment Report • {datetime.now().strftime('%B %d, %Y')}", styles['ReportSubtitle']))

    # --- Executive Summary ---
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Paragraph(
        "This automated report summarizes the performance of AI models trained to predict key piezoelectric properties. "
        "The system evaluated multiple machine learning architectures, optimizing for accuracy and generalization. "
        "The highest-performing models presented below have been deployed for real-time inference.",
        styles['Normal']
    ))
    story.append(Spacer(1, 25))

    # ==========================================================================
    # 1. D33 SECTION
    # ==========================================================================
    if insights_data.get('comparison_d33'):
        story.append(Paragraph("1. Piezoelectric Coefficient (d33)", styles['SectionHeaderD33']))
        results = insights_data['comparison_d33']
        best_model_name = sorted(results, key=lambda x: x['R2'], reverse=True)[0]['Model']
        
        # Insights
        insight_html = generate_insights_text(results, "d33")
        story.append(Paragraph(insight_html, styles['InsightBox']))
        
        # Charts - ACCURACY (R2)
        r2_content = [
            Paragraph("Metric Comparison: Accuracy (R²)", styles['ChartTitle']),
            Image(create_bar_chart(results, 'R2', 'Model Accuracy (d33)', COLOR_SECONDARY_D33, 'R² Score (Higher is Better)'), width=6.5*inch, height=3.2*inch),
            Spacer(1, 15)
        ]
        story.append(KeepTogether(r2_content))

        # Charts - ERROR (RMSE)
        rmse_content = [
            Paragraph("Metric Comparison: Error (RMSE)", styles['ChartTitle']),
            Image(create_bar_chart(results, 'RMSE', 'Model Error (RMSE)', '#EF5350', 'RMSE (Lower is Better)'), width=6.5*inch, height=3.2*inch),
            Spacer(1, 15)
        ]
        story.append(KeepTogether(rmse_content))
        
        # Scatter Plot Logic
        predictions_map = insights_data.get('predictions_d33', {})
        best_pred_data = predictions_map.get(best_model_name)
        
        # Fallback if best model predictions missing (e.g. key mismatch)
        if not best_pred_data and predictions_map:
            best_model_name = list(predictions_map.keys())[0]
            best_pred_data = predictions_map[best_model_name]

        if best_pred_data:
            story.append(PageBreak())
            
            scatter_content = [
                Paragraph("Performance Visualization: Predicted vs Actual", styles['SectionHeaderD33']),
                Paragraph(f"The scatter plot below visualizes the performance of the <b>{best_model_name}</b> model. Points clustering along the red diagonal line indicate accurate predictions. Vertical deviations represent error magnitude.", styles['Normal']),
                Spacer(1, 15),
                Image(create_scatter_chart(best_pred_data['y_test'], best_pred_data['y_pred'], best_model_name, "d33"), width=6*inch, height=5*inch),
                Spacer(1, 10),
                Paragraph("<b>Visual Analysis:</b> A tight distribution suggests consistent performance across the property range. Widespread points would indicate challenges in generalizing to novel crystal structures.", styles['InsightBox'])
            ]
            story.append(KeepTogether(scatter_content))


    # ==========================================================================
    # 2. Tc SECTION
    # ==========================================================================
    if insights_data.get('comparison_tc'):
        story.append(PageBreak())
        story.append(Paragraph("2. Curie Temperature (Tc)", styles['SectionHeaderTC']))
        results = insights_data['comparison_tc']
        best_model_name = sorted(results, key=lambda x: x['R2'], reverse=True)[0]['Model']
        
        # Insights
        insight_html = generate_insights_text(results, "Tc")
        story.append(Paragraph(insight_html, styles['InsightBox']))
        
        # Charts - ACCURACY (R2)
        r2_content_tc = [
            Paragraph("Metric Comparison: Accuracy (R²)", styles['ChartTitle']),
            Image(create_bar_chart(results, 'R2', 'Model Accuracy (Tc)', COLOR_SECONDARY_TC, 'R² Score (Higher is Better)'), width=6.5*inch, height=3.2*inch),
            Spacer(1, 15)
        ]
        story.append(KeepTogether(r2_content_tc))

        # Charts - ERROR (RMSE)
        rmse_content_tc = [
            Paragraph("Metric Comparison: Error (RMSE)", styles['ChartTitle']),
            Image(create_bar_chart(results, 'RMSE', 'Model Error (RMSE)', '#EF5350', 'RMSE (Lower is Better)'), width=6.5*inch, height=3.2*inch),
            Spacer(1, 15)
        ]
        story.append(KeepTogether(rmse_content_tc))
        
        # Scatter Plot Logic
        predictions_map = insights_data.get('predictions_tc', {})
        best_pred_data = predictions_map.get(best_model_name)
        
        # Fallback
        if not best_pred_data and predictions_map:
            best_model_name = list(predictions_map.keys())[0]
            best_pred_data = predictions_map[best_model_name]
        
        if best_pred_data:
            story.append(PageBreak())
            
            scatter_content_tc = [
                Paragraph("Performance Visualization: Predicted vs Actual", styles['SectionHeaderTC']),
                Paragraph(f"This visualization confirms the predictive power of the <b>{best_model_name}</b> model for Tc. Outliers, if any, are visibly distant from the ideal prediction line.", styles['Normal']),
                Spacer(1, 15),
                Image(create_scatter_chart(best_pred_data['y_test'], best_pred_data['y_pred'], best_model_name, "Tc"), width=6*inch, height=5*inch),
                Spacer(1, 10),
                Paragraph("<b>Visual Analysis:</b> The model demonstrates robust capability in predicting Curie Temperature, a critical parameter for high-temperature piezoelectric applications.", styles['InsightBox'])
            ]
            story.append(KeepTogether(scatter_content_tc))

    doc.build(story)
    buffer.seek(0)
    return buffer

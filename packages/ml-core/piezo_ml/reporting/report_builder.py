"""
Piezo.AI — Premium PDF Report Builder
========================================
Uses ReportLab to generate modern, professionally designed PDF reports with:
- Clean light theme with premium accent colors
- Piezo.AI branding (header/footer on every page)
- Embedded Matplotlib charts with proper aspect ratios
- Model performance tables with modern styling
- Prediction & usage insights
- Convergence charts per model
- Smart title-graph co-location (KeepTogether)

ARCHITECTURAL NOTE: This module lives in packages/ml-core/ but is called
from the dashboard service (apps/api/) via import. No FastAPI logic here.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from piezo_ml.reporting.chart_generator import (
    generate_convergence_chart,
    generate_model_performance_chart,
    generate_r2_rmse_chart,
    generate_target_distribution_chart,
    generate_usage_chart,
)
from piezo_ml.reporting.llm_insights import (
    generate_model_performance_insight,
    generate_material_prediction_insight,
)


# ── Brand Colors — Light Professional Theme ─────────────────
BRAND_INDIGO = colors.HexColor("#4F46E5")
BRAND_INDIGO_LIGHT = colors.HexColor("#EEF2FF")
BRAND_INDIGO_DARK = colors.HexColor("#3730A3")
BRAND_WHITE = colors.HexColor("#FFFFFF")
BRAND_BG = colors.HexColor("#FAFBFC")
BRAND_BORDER = colors.HexColor("#E2E8F0")
BRAND_BORDER_LIGHT = colors.HexColor("#F1F5F9")
BRAND_TEXT = colors.HexColor("#1E293B")
BRAND_TEXT_SECONDARY = colors.HexColor("#64748B")
BRAND_EMERALD = colors.HexColor("#10B981")
BRAND_EMERALD_LIGHT = colors.HexColor("#D1FAE5")
BRAND_AMBER = colors.HexColor("#F59E0B")
BRAND_AMBER_LIGHT = colors.HexColor("#FEF3C7")
BRAND_RED_LIGHT = colors.HexColor("#FEE2E2")

PAGE_W, PAGE_H = A4

# Available frame width (A4 - margins)
FRAME_W = PAGE_W - 40 * mm  # ~469.89 pts


class PiezoReportBuilder:
    """Builds premium Piezo.AI PDF reports."""

    def __init__(self):
        self.styles = self._create_styles()
        self._temp_files: list[str] = []

    def build(self, data: dict[str, Any], output_path: str) -> str:
        """Generate a complete PDF report."""
        doc = BaseDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=30 * mm,
            bottomMargin=25 * mm,
            title="Piezo.AI Model Performance Report",
            author="Piezo.AI v2.1.0",
            subject="AI-Driven Piezoelectric Material Discovery",
            creator="Piezo.AI Report Engine",
        )

        # Frame within margins
        frame = Frame(
            doc.leftMargin, doc.bottomMargin,
            doc.width, doc.height,
            id="main",
        )
        template = PageTemplate(
            id="main",
            frames=[frame],
            onPage=self._draw_page_chrome,
        )
        doc.addPageTemplates([template])

        # Build story (content)
        story = self._build_story(data)
        doc.build(story)

        # Cleanup temp chart files
        self._cleanup_temp_files()

        return output_path

    # ── Styles ────────────────────────────────────────────

    def _create_styles(self) -> dict[str, ParagraphStyle]:
        """Create premium typography styles — light theme."""
        base = getSampleStyleSheet()
        return {
            "title": ParagraphStyle(
                "PiezoTitle",
                parent=base["Heading1"],
                fontSize=24,
                textColor=BRAND_TEXT,
                spaceAfter=4 * mm,
                spaceBefore=2 * mm,
                fontName="Helvetica-Bold",
                leading=30,
            ),
            "subtitle": ParagraphStyle(
                "PiezoSubtitle",
                parent=base["Heading2"],
                fontSize=15,
                textColor=BRAND_INDIGO,
                spaceAfter=4 * mm,
                spaceBefore=8 * mm,
                fontName="Helvetica-Bold",
                leading=20,
            ),
            "section": ParagraphStyle(
                "PiezoSection",
                parent=base["Heading3"],
                fontSize=12,
                textColor=BRAND_TEXT,
                spaceAfter=3 * mm,
                spaceBefore=5 * mm,
                fontName="Helvetica-Bold",
                leading=16,
            ),
            "body": ParagraphStyle(
                "PiezoBody",
                parent=base["Normal"],
                fontSize=10,
                textColor=BRAND_TEXT,
                spaceAfter=2 * mm,
                leading=14,
                fontName="Helvetica",
            ),
            "small": ParagraphStyle(
                "PiezoSmall",
                parent=base["Normal"],
                fontSize=8,
                textColor=BRAND_TEXT_SECONDARY,
                leading=11,
                fontName="Helvetica",
            ),
            "caption": ParagraphStyle(
                "PiezoCaption",
                parent=base["Normal"],
                fontSize=8,
                textColor=BRAND_TEXT_SECONDARY,
                spaceAfter=2 * mm,
                spaceBefore=1 * mm,
                leading=10,
                fontName="Helvetica-Oblique",
                alignment=TA_CENTER,
            ),
        }

    # ── Page Chrome (Header/Footer) ─────────────────────

    def _draw_page_chrome(self, canvas, doc):
        """Draw branded header and footer on every page."""
        canvas.saveState()

        # Header bar — indigo gradient feel
        canvas.setFillColor(BRAND_INDIGO)
        canvas.rect(0, PAGE_H - 18 * mm, PAGE_W, 18 * mm, fill=1, stroke=0)

        # Subtle lighter accent strip
        canvas.setFillColor(BRAND_INDIGO_DARK)
        canvas.rect(0, PAGE_H - 18 * mm, PAGE_W, 0.5 * mm, fill=1, stroke=0)

        # Header text
        canvas.setFillColor(BRAND_WHITE)
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(20 * mm, PAGE_H - 12.5 * mm, "Piezo.AI")

        canvas.setFont("Helvetica", 8.5)
        canvas.drawRightString(
            PAGE_W - 20 * mm, PAGE_H - 12.5 * mm,
            "AI-Driven Piezoelectric Material Discovery"
        )

        # Footer
        canvas.setStrokeColor(BRAND_BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(20 * mm, 16 * mm, PAGE_W - 20 * mm, 16 * mm)

        canvas.setFillColor(BRAND_TEXT_SECONDARY)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(
            20 * mm, 10 * mm,
            f"Generated by Piezo.AI v2.1.0 — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        canvas.drawRightString(
            PAGE_W - 20 * mm, 10 * mm,
            f"Page {canvas.getPageNumber()}"
        )

        canvas.restoreState()

    # ── Story Builder ────────────────────────────────────

    def _build_story(self, data: dict[str, Any]) -> list:
        """Assemble the full report content."""
        story = []
        options = data.get("options", {})
        models = data.get("models", [])
        predictions = data.get("predictions", [])
        stats = data.get("stats", {})

        # Title
        story.append(Spacer(1, 8 * mm))
        story.append(Paragraph("Model Performance Report", self.styles["title"]))
        story.append(Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            self.styles["small"],
        ))
        story.append(Spacer(1, 6 * mm))

        # System overview
        story.extend(self._build_stats_section(stats))

        # Model summary table
        if models:
            story.append(Spacer(1, 4 * mm))
            story.extend(self._build_model_table(models))

        # R²/RMSE charts — keep title and chart together
        if options.get("include_r2_rmse", True) and models:
            story.append(Spacer(1, 4 * mm))
            story.extend(self._build_r2_rmse_section(models))

        # Target distribution
        if models:
            story.extend(self._build_target_distribution(models))

        # Model performance overview
        if models:
            story.extend(self._build_performance_overview(models))

        # Convergence charts
        if models:
            story.extend(self._build_convergence_section(data))

        # AI insight (LLM-powered analysis)
        if options.get("include_ai_insight", False):
            llm_config = data.get("llm_config", {})
            story.extend(self._build_ai_insight_section(
                models, predictions, stats, llm_config,
            ))

        # Prediction insights
        if predictions and options.get("include_material_insight", False):
            story.extend(self._build_prediction_section(predictions))

        # Usage predictions
        if predictions and options.get("include_material_insight", False):
            story.extend(self._build_usage_section(predictions))

        return story

    # ── Section: System Stats ────────────────────────────

    def _build_stats_section(self, stats: dict) -> list:
        """Build system overview stats."""
        elements = []
        elements.append(Paragraph("System Overview", self.styles["subtitle"]))
        elements.append(Paragraph(
            "A snapshot of the current Piezo.AI platform state, including datasets loaded, "
            "models trained, predictions made, and database utilization.",
            self.styles["body"],
        ))
        elements.append(Spacer(1, 2 * mm))

        stat_data = [
            [
                Paragraph("<b>Metric</b>", self._cell_style(BRAND_WHITE, 9, True)),
                Paragraph("<b>Value</b>", self._cell_style(BRAND_WHITE, 9, True, TA_CENTER)),
            ],
            ["Datasets", str(stats.get("dataset_count", 0))],
            ["Total Material Rows", str(stats.get("total_material_rows", 0))],
            ["Trained Models", str(stats.get("trained_model_count", 0))],
            ["Predictions Made", str(stats.get("prediction_count", 0))],
            ["Training Jobs", str(stats.get("training_job_count", 0))],
            ["Database Size", f"{stats.get('db_size_mb', 0):.1f} MB"],
        ]

        table = Table(stat_data, colWidths=[160, 140])
        table.setStyle(self._premium_table_style(len(stat_data)))
        elements.append(table)
        return elements

    # ── Section: Model Table ─────────────────────────────

    def _build_model_table(self, models: list[dict]) -> list:
        """Build model summary table with proper column widths to prevent overlap."""
        elements = []
        elements.append(Paragraph("Trained Models Summary", self.styles["subtitle"]))
        elements.append(Paragraph(
            "Overview of all trained ML models with their target property, algorithm, "
            "R\u00b2 goodness-of-fit score, RMSE prediction error, and training/test split sizes.",
            self.styles["body"],
        ))
        elements.append(Spacer(1, 2 * mm))

        # Use Paragraph objects for wrappable cells
        header = [
            Paragraph("<b>Name</b>", self._cell_style(BRAND_WHITE, 8, True)),
            Paragraph("<b>Target</b>", self._cell_style(BRAND_WHITE, 8, True, TA_CENTER)),
            Paragraph("<b>Algorithm</b>", self._cell_style(BRAND_WHITE, 8, True, TA_CENTER)),
            Paragraph("<b>R²</b>", self._cell_style(BRAND_WHITE, 8, True, TA_CENTER)),
            Paragraph("<b>RMSE</b>", self._cell_style(BRAND_WHITE, 8, True, TA_CENTER)),
            Paragraph("<b>Train</b>", self._cell_style(BRAND_WHITE, 8, True, TA_CENTER)),
            Paragraph("<b>Test</b>", self._cell_style(BRAND_WHITE, 8, True, TA_CENTER)),
        ]
        rows = [header]

        target_labels = {"d33": "d33", "tc": "Tc", "vickers_hardness": "HV"}

        for m in models:
            r2 = m.get("r2_score", 0)
            r2_color = "#10B981" if r2 >= 0.8 else ("#F59E0B" if r2 >= 0.5 else "#EF4444")
            rows.append([
                Paragraph(m.get("display_name", "?")[:28], self._cell_style(BRAND_TEXT, 8)),
                Paragraph(target_labels.get(m.get("target", ""), m.get("target", "?")),
                         self._cell_style(BRAND_TEXT, 8, alignment=TA_CENTER)),
                Paragraph(m.get("algorithm", "?"),
                         self._cell_style(BRAND_TEXT, 8, alignment=TA_CENTER)),
                Paragraph(f"<font color='{r2_color}'><b>{r2:.4f}</b></font>",
                         self._cell_style(BRAND_TEXT, 8, alignment=TA_CENTER)),
                Paragraph(f"{m.get('rmse', 0):.2f}",
                         self._cell_style(BRAND_TEXT, 8, alignment=TA_CENTER)),
                Paragraph(str(m.get("n_train_samples", 0)),
                         self._cell_style(BRAND_TEXT, 8, alignment=TA_CENTER)),
                Paragraph(str(m.get("n_test_samples", 0)),
                         self._cell_style(BRAND_TEXT, 8, alignment=TA_CENTER)),
            ])

        # Wider column layout to prevent text overlap
        col_widths = [130, 50, 80, 55, 55, 45, 45]
        table = Table(rows, colWidths=col_widths)
        table.setStyle(self._premium_table_style(len(rows)))
        elements.append(table)
        return elements

    # ── Section: R²/RMSE Charts ──────────────────────────

    def _build_r2_rmse_section(self, models: list[dict]) -> list:
        """Build R²/RMSE chart section — KeepTogether for title + chart."""
        chart_path = generate_r2_rmse_chart(models)
        if not chart_path or not os.path.exists(chart_path):
            return []

        self._temp_files.append(chart_path)
        img = self._fit_image(chart_path, max_width=FRAME_W, max_height=280)

        block = [
            Paragraph("R² and RMSE Comparison", self.styles["subtitle"]),
            Paragraph(
                "Side-by-side comparison of R² (coefficient of determination) and RMSE "
                "(root mean square error) across all trained models. Higher R² and lower RMSE "
                "indicate better model accuracy and reliability for material property predictions.",
                self.styles["body"],
            ),
            Spacer(1, 2 * mm),
            img,
            Spacer(1, 4 * mm),
        ]
        return [KeepTogether(block)]

    # ── Section: Target Distribution ─────────────────────

    def _build_target_distribution(self, models: list[dict]) -> list:
        """Build target distribution donut chart."""
        chart_path = generate_target_distribution_chart(models)
        if not chart_path or not os.path.exists(chart_path):
            return []

        self._temp_files.append(chart_path)
        img = self._fit_image(chart_path, max_width=320, max_height=280)

        block = [
            Paragraph("Model Target Distribution", self.styles["subtitle"]),
            Paragraph(
                "Distribution of trained models across target properties (d\u2083\u2083, Tc, Hardness). "
                "Shows the proportion of models dedicated to predicting each material property, "
                "indicating research focus and coverage balance.",
                self.styles["body"],
            ),
            Spacer(1, 2 * mm),
            img,
            Spacer(1, 4 * mm),
        ]
        return [KeepTogether(block)]

    # ── Section: Performance Overview ────────────────────

    def _build_performance_overview(self, models: list[dict]) -> list:
        """Build model performance horizontal bar chart."""
        chart_path = generate_model_performance_chart(models)
        if not chart_path or not os.path.exists(chart_path):
            return []

        self._temp_files.append(chart_path)
        # Dynamic height: more models = taller chart, but cap it
        chart_height = min(400, max(180, len(models) * 40 + 60))
        img = self._fit_image(chart_path, max_width=FRAME_W, max_height=chart_height)

        title = Paragraph("Performance Overview", self.styles["subtitle"])
        desc = Paragraph(
            "Visual comparison of model performance via R² scores. The dashed line at 0.8 "
            "represents the threshold for reliable predictive quality. Models exceeding this "
            "threshold are considered suitable for guiding experimental material synthesis.",
            self.styles["body"],
        )

        # For small charts, keep title + chart together; for large, let flow
        if len(models) <= 6:
            return [KeepTogether([title, desc, Spacer(1, 2 * mm), img, Spacer(1, 4 * mm)])]
        else:
            return [title, desc, Spacer(1, 2 * mm), img, Spacer(1, 4 * mm)]

    # ── Section: Convergence Charts ──────────────────────

    def _build_convergence_section(self, data: dict) -> list:
        """Build convergence charts for models that have convergence data."""
        elements = []
        convergence_data = data.get("convergence_data", {})

        if not convergence_data:
            return elements

        elements.append(PageBreak())
        elements.append(Paragraph("Training Convergence", self.styles["subtitle"]))
        elements.append(Paragraph(
            "Convergence curves showing training loss/metric progression over iterations.",
            self.styles["body"],
        ))

        chart_path = generate_convergence_chart(convergence_data)
        if chart_path and os.path.exists(chart_path):
            self._temp_files.append(chart_path)
            img = self._fit_image(chart_path, max_width=FRAME_W, max_height=300)
            elements.append(img)
            elements.append(Spacer(1, 4 * mm))

        return elements

    # ── Section: AI Insight ────────────────────────────────

    def _build_ai_insight_section(
        self, models: list[dict], predictions: list[dict],
        stats: dict, llm_config: dict,
    ) -> list:
        """Build AI-powered insight section using configured LLM."""
        elements = []

        provider = llm_config.get("provider", "google")
        api_key = llm_config.get("api_key", "")
        model_name = llm_config.get("model", "gemini-2.0-flash")
        temperature = llm_config.get("temperature", 0.1)
        max_tokens = llm_config.get("max_tokens", 2048)

        elements.append(PageBreak())
        elements.append(Paragraph("AI Performance Analysis", self.styles["subtitle"]))

        # Info box about AI source
        if api_key:
            source_text = f"Generated by {provider}/{model_name}"
        else:
            source_text = "Rule-based analysis (LLM not configured)"

        info_style = ParagraphStyle(
            "AIInfoBox",
            fontSize=8,
            textColor=BRAND_INDIGO,
            backColor=BRAND_INDIGO_LIGHT,
            borderPadding=(4, 6, 4, 6),
            leading=11,
            fontName="Helvetica-Oblique",
            spaceAfter=4 * mm,
        )
        elements.append(Paragraph(source_text, info_style))

        # Generate model performance insight
        perf_text = generate_model_performance_insight(
            models, stats,
            provider=provider, api_key=api_key, model_name=model_name,
            temperature=temperature, max_tokens=max_tokens,
        )

        # Split into paragraphs for better formatting
        for para in perf_text.split("\n\n"):
            para = para.strip()
            if para:
                elements.append(Paragraph(para, self.styles["body"]))
                elements.append(Spacer(1, 2 * mm))

        # Material prediction insight (if predictions available)
        if predictions:
            elements.append(Spacer(1, 4 * mm))
            elements.append(Paragraph("AI Material Insight", self.styles["subtitle"]))

            mat_text = generate_material_prediction_insight(
                predictions,
                provider=provider, api_key=api_key, model_name=model_name,
                temperature=temperature, max_tokens=max_tokens,
            )

            for para in mat_text.split("\n\n"):
                para = para.strip()
                if para:
                    elements.append(Paragraph(para, self.styles["body"]))
                    elements.append(Spacer(1, 2 * mm))

        return elements

    # ── Section: Prediction Insights ─────────────────────

    def _build_prediction_section(self, predictions: list[dict]) -> list:
        """Build prediction insights section with property values and use cases."""
        elements = []
        elements.append(PageBreak())
        elements.append(Paragraph("Prediction Insights", self.styles["subtitle"]))
        elements.append(Paragraph(
            "Predicted material properties for selected formulas, including d\u2083\u2083 piezoelectric "
            "coefficient, Curie temperature (Tc), Vickers hardness, and the top suggested use case "
            "based on the combination of predicted properties.",
            self.styles["body"],
        ))
        elements.append(Spacer(1, 2 * mm))

        # Group predictions by formula to merge d33/tc/hardness
        merged = self._merge_predictions_by_formula(predictions)

        header = [
            Paragraph("<b>Formula</b>", self._cell_style(BRAND_WHITE, 7, True)),
            Paragraph("<b>d<sub>33</sub></b>", self._cell_style(BRAND_WHITE, 7, True, TA_CENTER)),
            Paragraph("<b>Tc</b>", self._cell_style(BRAND_WHITE, 7, True, TA_CENTER)),
            Paragraph("<b>HV</b>", self._cell_style(BRAND_WHITE, 7, True, TA_CENTER)),
            Paragraph("<b>Status</b>", self._cell_style(BRAND_WHITE, 7, True, TA_CENTER)),
        ]
        rows = [header]
        for p in merged[:25]:
            d33_val = f"{p['d33_predicted']:.1f}" if p.get("d33_predicted") is not None else "\u2014"
            tc_val = f"{p['tc_predicted']:.1f}" if p.get("tc_predicted") is not None else "\u2014"
            hv_val = f"{p['hardness_predicted']:.1f}" if p.get("hardness_predicted") is not None else "\u2014"
            status = p.get("prediction_status", "?")
            status_color = "#10B981" if status == "success" else "#EF4444"

            rows.append([
                Paragraph(p.get("formula", "?")[:30], self._cell_style(BRAND_TEXT, 7)),
                Paragraph(d33_val, self._cell_style(BRAND_TEXT, 7, alignment=TA_CENTER)),
                Paragraph(tc_val, self._cell_style(BRAND_TEXT, 7, alignment=TA_CENTER)),
                Paragraph(hv_val, self._cell_style(BRAND_TEXT, 7, alignment=TA_CENTER)),
                Paragraph(f"<font color='{status_color}'>{status}</font>",
                         self._cell_style(BRAND_TEXT, 7, alignment=TA_CENTER)),
            ])

        col_widths = [150, 70, 70, 70, 80]
        table = Table(rows, colWidths=col_widths)
        table.setStyle(self._premium_table_style(len(rows)))
        elements.append(table)
        return elements

    def _merge_predictions_by_formula(self, predictions: list[dict]) -> list[dict]:
        """Merge multiple prediction rows for the same formula into one."""
        merged: dict[str, dict] = {}
        for p in predictions:
            f = p.get("formula", "")
            if f not in merged:
                merged[f] = dict(p)
            else:
                # Merge property values
                if p.get("d33_predicted") is not None and merged[f].get("d33_predicted") is None:
                    merged[f]["d33_predicted"] = p["d33_predicted"]
                if p.get("tc_predicted") is not None and merged[f].get("tc_predicted") is None:
                    merged[f]["tc_predicted"] = p["tc_predicted"]
                if p.get("hardness_predicted") is not None and merged[f].get("hardness_predicted") is None:
                    merged[f]["hardness_predicted"] = p["hardness_predicted"]
        return list(merged.values())

    # ── Section: Usage Predictions ────────────────────────

    def _build_usage_section(self, predictions: list[dict]) -> list:
        """Build usage predictions section showing use-case fit for materials."""
        elements = []
        # Collect usage data from predictions
        usage_items = []
        for p in predictions:
            usage = p.get("usage_predictions", {})
            if usage and isinstance(usage, dict):
                recs = usage.get("recommendations", [])
                for r in recs:
                    usage_items.append({
                        "formula": p.get("formula", "?"),
                        "use_case": r.get("use_case", "?"),
                        "score": r.get("score", r.get("confidence", 0) * 100),
                        "tier": r.get("tier_label", r.get("tier", "?")),
                    })

        if not usage_items:
            return elements

        elements.append(Spacer(1, 6 * mm))
        elements.append(Paragraph("Material Usage Predictions", self.styles["subtitle"]))
        elements.append(Paragraph(
            "Use-case suitability analysis for predicted materials with tier classification.",
            self.styles["body"],
        ))

        header = [
            Paragraph("<b>Formula</b>", self._cell_style(BRAND_WHITE, 8, True)),
            Paragraph("<b>Use Case</b>", self._cell_style(BRAND_WHITE, 8, True)),
            Paragraph("<b>Score</b>", self._cell_style(BRAND_WHITE, 8, True, TA_CENTER)),
            Paragraph("<b>Fit</b>", self._cell_style(BRAND_WHITE, 8, True, TA_CENTER)),
        ]
        rows = [header]
        for item in usage_items[:30]:
            score = item.get("score", 0)
            if score >= 70:
                score_color = "#10B981"
            elif score >= 45:
                score_color = "#F59E0B"
            else:
                score_color = "#64748B"

            rows.append([
                Paragraph(item.get("formula", "?")[:25], self._cell_style(BRAND_TEXT, 8)),
                Paragraph(item.get("use_case", "?")[:30], self._cell_style(BRAND_TEXT, 8)),
                Paragraph(f"<font color='{score_color}'><b>{score:.0f}</b></font>",
                         self._cell_style(BRAND_TEXT, 8, alignment=TA_CENTER)),
                Paragraph(item.get("tier", "?"),
                         self._cell_style(BRAND_TEXT, 8, alignment=TA_CENTER)),
            ])

        col_widths = [120, 160, 60, 100]
        table = Table(rows, colWidths=col_widths)
        table.setStyle(self._premium_table_style(len(rows)))
        elements.append(table)

        # Generate usage chart if enough data
        if len(usage_items) >= 2:
            chart_path = generate_usage_chart(usage_items)
            if chart_path and os.path.exists(chart_path):
                self._temp_files.append(chart_path)
                elements.append(Spacer(1, 4 * mm))
                img = self._fit_image(chart_path, max_width=FRAME_W, max_height=300)
                elements.append(img)

        return elements

    # ── Helpers ───────────────────────────────────────────

    def _fit_image(self, path: str, max_width: float, max_height: float) -> Image:
        """Create an Image flowable that fits within bounds, preserving aspect ratio.
        
        IMPORTANT: max_height is also capped to the usable frame height (673pt for A4)
        to prevent ReportLab 'too large on page' errors.
        """
        from reportlab.lib.utils import ImageReader
        
        # Frame height for A4 with our margins (30mm top + 25mm bottom)
        FRAME_H = PAGE_H - 30 * mm - 25 * mm  # ~673.98 pts
        max_height = min(max_height, FRAME_H - 60)  # Leave space for title/caption
        
        try:
            reader = ImageReader(path)
            iw, ih = reader.getSize()
            # Scale to fit within max dimensions
            scale_w = max_width / iw
            scale_h = max_height / ih
            scale = min(scale_w, scale_h, 1.0)  # Don't upscale
            return Image(path, width=iw * scale, height=ih * scale)
        except Exception:
            # Fallback: use fixed dimensions
            return Image(path, width=min(max_width, 400), height=min(max_height, 200))

    def _cell_style(
        self, text_color, size: int = 9, bold: bool = False,
        alignment: int = TA_LEFT,
    ) -> ParagraphStyle:
        """Create a cell-level paragraph style."""
        name = f"Cell_{id(text_color)}_{size}_{bold}_{alignment}"
        font = "Helvetica-Bold" if bold else "Helvetica"
        return ParagraphStyle(
            name, fontName=font, fontSize=size,
            textColor=text_color, alignment=alignment,
            leading=size + 4,
        )

    def _premium_table_style(self, n_rows: int) -> TableStyle:
        """Create a premium table style with rounded feel, alternating rows."""
        return TableStyle([
            # Header
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_INDIGO),
            ("TEXTCOLOR", (0, 0), (-1, 0), BRAND_WHITE),
            # Alternating rows
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BRAND_WHITE, BRAND_BG]),
            # Grid
            ("GRID", (0, 0), (-1, 0), 0, BRAND_INDIGO),  # No grid on header
            ("LINEBELOW", (0, 0), (-1, 0), 1, BRAND_INDIGO_DARK),
            ("LINEBELOW", (0, 1), (-1, -2), 0.3, BRAND_BORDER),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, BRAND_BORDER),
            # Outer border
            ("BOX", (0, 0), (-1, -1), 0.5, BRAND_BORDER),
            # Padding
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 1), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            # Vertical alignment
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])

    # ── Cleanup ──────────────────────────────────────────

    def _cleanup_temp_files(self):
        """Remove temporary chart files."""
        for f in self._temp_files:
            try:
                os.unlink(f)
            except Exception:
                pass
        self._temp_files.clear()

"""
Comprehensive generator for X-Maintain IEEE Research Paper.
Generates:
1. X_Maintain_IEEE_Research_Paper.pdf (ReportLab 2-column IEEE layout, exactly 8 pages, beautifully balanced)
2. X_Maintain_IEEE_Research_Paper.docx (python-docx 2-column IEEE layout)
3. X_Maintain_IEEE_Research_Paper.tex (LaTeX IEEEtran format)
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# ReportLab imports
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm, mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    Table, TableStyle, Image, KeepTogether, FrameBreak, PageBreak, NextPageTemplate
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT

# docx imports
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

# PyMuPDF for visual inspection
import fitz

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

print("Loading project metrics and data...")
with open(os.path.join(BASE_DIR, 'artifacts', 'metrics', 'metrics.json')) as f:
    metrics = json.load(f)

research_df = pd.read_csv(os.path.join(BASE_DIR, 'artifacts', 'metrics', 'research_experiments_summary.csv'))
importance_df = pd.read_csv(os.path.join(BASE_DIR, 'artifacts', 'metrics', 'feature_importance.csv'))
with open(os.path.join(BASE_DIR, 'artifacts', 'explanations', 'explanation_report.json')) as f:
    explanation_report = json.load(f)

# =========================================================================
# 1. REPORTLAB PDF GENERATION
# =========================================================================

class IEEENumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_ieee_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_ieee_header_footer(self, total_pages):
        self.saveState()
        self.setFont("Times-Roman", 8)
        self.setFillColor(colors.HexColor("#475569"))
        
        # Header (Pages 2+)
        if self._pageNumber > 1:
            if self._pageNumber % 2 == 0:
                self.drawString(36, 812, "IEEE TRANSACTIONS ON INDUSTRIAL INFORMATICS (SPECIAL SECTION ON EXPLAINABLE AI)")
            else:
                self.drawRightString(595 - 36, 812, "REDDY: X-MAINTAIN EXPLAINABLE MACHINE LEARNING FRAMEWORK FOR PREDICTIVE MAINTENANCE")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(36, 806, 595 - 36, 806)

        # Footer (All pages)
        footer_text = f"{self._pageNumber}"
        self.drawCentredString(595 / 2.0, 24, footer_text)
        self.restoreState()

def build_pdf():
    print("Building ReportLab PDF document...")
    pdf_path = os.path.join(BASE_DIR, "X_Maintain_IEEE_Research_Paper.pdf")

    PAGE_W, PAGE_H = A4 # 595.27 x 841.89 pt
    MARGIN_X = 36
    MARGIN_BOTTOM = 36
    PRINT_W = PAGE_W - 2 * MARGIN_X # 523.27 pt
    COL_GAP = 16
    COL_W = (PRINT_W - COL_GAP) / 2 # 253.63 pt

    doc = BaseDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    # First Page Frames:
    # Header frame for Title, Authors, Abstract, Index Terms
    # Measured exact height: ~275 pt. Setting HEADER_H = 285 pt ensures zero overflow into columns!
    HEADER_H = 285
    frame_top = Frame(
        MARGIN_X, PAGE_H - 36 - HEADER_H, PRINT_W, HEADER_H,
        id='F_header', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0
    )
    P1_COL_H = PAGE_H - 36 - HEADER_H - MARGIN_BOTTOM - 8
    frame_p1_c1 = Frame(
        MARGIN_X, MARGIN_BOTTOM, COL_W, P1_COL_H,
        id='F_p1_c1', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0
    )
    frame_p1_c2 = Frame(
        MARGIN_X + COL_W + COL_GAP, MARGIN_BOTTOM, COL_W, P1_COL_H,
        id='F_p1_c2', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0
    )

    # Standard Two-Column Frames:
    NORM_H = PAGE_H - 36 - MARGIN_BOTTOM - 20 # 750 pt
    frame_norm_c1 = Frame(
        MARGIN_X, MARGIN_BOTTOM, COL_W, NORM_H,
        id='F_norm_c1', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0
    )
    frame_norm_c2 = Frame(
        MARGIN_X + COL_W + COL_GAP, MARGIN_BOTTOM, COL_W, NORM_H,
        id='F_norm_c2', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0
    )

    page_p1 = PageTemplate(id='FirstPage', frames=[frame_top, frame_p1_c1, frame_p1_c2])
    page_norm = PageTemplate(id='NormPage', frames=[frame_norm_c1, frame_norm_c2])
    doc.addPageTemplates([page_p1, page_norm])

    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        'IEEETitle',
        fontName='Times-Bold',
        fontSize=17,
        leading=21,
        alignment=TA_CENTER,
        spaceAfter=6,
        textColor=colors.HexColor("#0f172a")
    )

    style_author = ParagraphStyle(
        'IEEEAuthor',
        fontName='Times-Bold',
        fontSize=9.5,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1e293b")
    )

    style_affil = ParagraphStyle(
        'IEEEAffil',
        fontName='Times-Roman',
        fontSize=8,
        leading=10.5,
        alignment=TA_CENTER,
        spaceAfter=8,
        textColor=colors.HexColor("#334155")
    )

    style_abstract = ParagraphStyle(
        'IEEEAbstract',
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#0f172a")
    )

    style_index = ParagraphStyle(
        'IEEEIndex',
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        alignment=TA_JUSTIFY,
        spaceAfter=2,
        textColor=colors.HexColor("#0f172a")
    )

    style_sec = ParagraphStyle(
        'IEEESection',
        fontName='Times-Bold',
        fontSize=9.5,
        leading=13,
        alignment=TA_CENTER,
        spaceBefore=8,
        spaceAfter=3,
        textColor=colors.HexColor("#0f172a")
    )

    style_subsec = ParagraphStyle(
        'IEEESubsection',
        fontName='Times-BoldItalic',
        fontSize=8.5,
        leading=11.5,
        alignment=TA_LEFT,
        spaceBefore=5,
        spaceAfter=2,
        textColor=colors.HexColor("#1e293b")
    )

    style_body = ParagraphStyle(
        'IEEEBody',
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        alignment=TA_JUSTIFY,
        spaceAfter=3.5,
        textColor=colors.HexColor("#0f172a"),
        firstLineIndent=10
    )

    style_body_noindent = ParagraphStyle(
        'IEEEBodyNoIndent',
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        alignment=TA_JUSTIFY,
        spaceAfter=3.5,
        textColor=colors.HexColor("#0f172a")
    )

    style_caption = ParagraphStyle(
        'IEEECaption',
        fontName='Times-Roman',
        fontSize=7.5,
        leading=9.5,
        alignment=TA_CENTER,
        spaceBefore=3,
        spaceAfter=6,
        textColor=colors.HexColor("#1e293b")
    )

    style_table_title = ParagraphStyle(
        'IEEETableTitle',
        fontName='Times-Bold',
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        spaceBefore=5,
        spaceAfter=2,
        textColor=colors.HexColor("#0f172a")
    )

    style_ref = ParagraphStyle(
        'IEEERef',
        fontName='Times-Roman',
        fontSize=7.2,
        leading=9.2,
        alignment=TA_JUSTIFY,
        spaceAfter=2.5,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=12,
        firstLineIndent=-12
    )

    style_equation = ParagraphStyle(
        'IEEEEquation',
        fontName='Times-Italic',
        fontSize=8,
        leading=10.5,
        alignment=TA_CENTER,
        spaceBefore=3,
        spaceAfter=3,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # Transition to NormPage on page 2+
    story.append(NextPageTemplate('NormPage'))

    # -------------------------------------------------------------
    # HEADER BLOCK (Page 1 Full-Width Frame)
    # -------------------------------------------------------------
    story.append(Paragraph("X-Maintain: An Explainable Machine Learning Framework for Predictive Maintenance and Machine Failure Prediction", style_title))
    story.append(Paragraph("Syagamreddy Gopi Adithya Vardhan Reddy", style_author))
    story.append(Paragraph("Department of Computer Science and Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu 632014, India<br/>(e-mail: gopi.adithya2022@vitstudent.ac.in)", style_affil))

    abstract_text = (
        "<b><i>Abstract</i>—Industrial predictive maintenance (PdM) relies on multivariate sensor streams to anticipate equipment "
        "failure before catastrophic downtime occurs. However, contemporary high-performance machine learning ensembles operate as opaque black boxes, "
        "creating an operational trust barrier for factory engineers who require actionable root-cause insights before halting costly production spindles. "
        "This paper presents X-Maintain, an end-to-end explainable artificial intelligence (XAI) framework for predictive maintenance. Using the benchmark "
        "AI4I 2020 Predictive Maintenance Dataset (10,000 milling operations), we enforce strict target leakage prevention by excluding failure-mode indicators "
        "(TWF, HDF, PWF, OSF, RNF) and arbitrary serial keys, yielding an eight-feature operational input matrix. We benchmark three classifier families "
        "— Logistic Regression, Random Forest, and Extreme Gradient Boosting (XGBoost) — and systematically evaluate four class-imbalance mitigation strategies "
        "under severe 28.5:1 class disparity. Model selection is governed by a domain-grounded utility metric (0.4&times;Recall + 0.3&times;F1 + 0.3&times;ROC-AUC). "
        "On a held-out test split (<i>N</i> = 2,001), cost-sensitive XGBoost achieves 98.20% accuracy, 73.53% precision, 73.53% failure recall, 0.9724 ROC-AUC, "
        "and 0.7934 PR-AUC. Model decisions are interpreted using TreeSHAP, revealing that Torque (mean |SHAP| = 3.74) and Tool Wear (mean |SHAP| = 3.06) dominate "
        "failure predictions. Local waterfall attributions, a deterministic hallucination-free natural language explanation generator, a model-based what-if "
        "sensitivity engine, and an enterprise Streamlit monitoring dashboard complete the validated architecture, backed by 20 passed unit tests.</b>"
    )
    story.append(Paragraph(abstract_text, style_abstract))

    index_text = (
        "<b><i>Index Terms</i>—Predictive maintenance, Explainable Artificial Intelligence (XAI), SHapley Additive exPlanations (SHAP), "
        "machine failure prediction, XGBoost, class imbalance, TreeExplainer, industrial IoT, model-based sensitivity analysis.</b>"
    )
    story.append(Spacer(1, 3))
    story.append(Paragraph(index_text, style_index))

    # Advance from Header frame to Column 1 Frame on Page 1
    story.append(FrameBreak())

    # -------------------------------------------------------------
    # SECTION I: INTRODUCTION
    # -------------------------------------------------------------
    story.append(Paragraph("I. INTRODUCTION", style_sec))
    story.append(Paragraph(
        "INDUSTRIAL manufacturing relies heavily on computerized numerical control (CNC) milling machinery, automated tooling spindles, and multi-axis machining centers. "
        "Equipment breakdowns incur severe financial and operational losses, costing industrial facilities an estimated $50 billion annually in unplanned downtime [3], [4]. "
        "Historically, manufacturing facilities adopted reactive 'run-to-failure' strategies or rigid time-based preventative schedules [3]. While reactive repair creates "
        "unacceptable emergency outages, preventative maintenance frequently causes premature component replacement, discarding functional cutting heads with substantial useful life remaining [4].",
        style_body
    ))
    story.append(Paragraph(
        "Predictive maintenance (PdM) leverages multivariate sensor telemetry—such as spindle torque, cutting contact duration, rotational velocities, and thermal dissipation gradients—to dynamically "
        "forecast impending failure states [1], [5]. However, real-world industrial adoption faces the critical 'black-box dilemma' [6], [7]. Advanced deep learning and tree ensemble algorithms output "
        "probabilistic risk scores without exposing the underlying physical drivers. If an autonomous model flags an alert without physical attribution, plant operators cannot ascertain whether the alarm "
        "stems from thermal exhaustion, mechanical overstrain, or electrical fluctuation [7]. Under high-stakes operational pressure, uninterpretable alerts lead to either costly unnecessary line halts or "
        "dangerous alarm dismissal.",
        style_body
    ))
    story.append(Paragraph(
        "Furthermore, predictive maintenance is governed by profound misclassification cost asymmetry. Missing an actual machine failure (False Negative) leads to catastrophic spindle destruction, ruined workpieces, "
        "and collateral assembly line stoppage, with remediation costs often exceeding $10,000 to $100,000 [3]. Conversely, a false warning (False Positive) incurs only a brief 10-minute diagnostic inspection ($100–$500). "
        "Consequently, standard classification accuracy is an inappropriate metric under severe minority failure distributions, demanding optimization focused heavily on Failure Recall [6].",
        style_body
    ))
    story.append(Paragraph(
        "To address these challenges, we present <i>X-Maintain</i>, an Explainable AI predictive maintenance framework that pairs high-recall gradient boosting with exact game-theoretic Shapley attributions (TreeSHAP) [2], [12]. "
        "The primary verified contributions of this work are:",
        style_body
    ))
    story.append(Paragraph("1) <i>Leakage-Aware Preprocessing:</i> A rigorous data pipeline that purges downstream failure modes (TWF, HDF, PWF, OSF, RNF) and primary serial keys to guarantee academic and operational integrity.", style_body_noindent))
    story.append(Paragraph("2) <i>Empirical Imbalance Benchmarking:</i> A comparative evaluation across baseline Logistic Regression, Random Forest, and XGBoost under cost-sensitive weighting, SMOTE, and validation-tuned decision thresholds.", style_body_noindent))
    story.append(Paragraph("3) <i>Multi-Objective Model Selection:</i> A weighted utility formulation (0.4&times;Recall + 0.3&times;F1 + 0.3&times;ROC-AUC) optimizing minority failure capture while suppressing alarm fatigue.", style_body_noindent))
    story.append(Paragraph("4) <i>Dual-Scale TreeSHAP Explainability:</i> Exact global attribution across the dataset manifold and local waterfall decomposition for individual machine predictions.", style_body_noindent))
    story.append(Paragraph("5) <i>Model-Based What-If Analysis:</i> A counterfactual parameter perturbation engine allowing operators to simulate risk reduction under simulated operational adjustments without claiming physical causality.", style_body_noindent))
    story.append(Paragraph("6) <i>Production Deployment & Verification:</i> An enterprise-grade 7-view Streamlit industrial monitoring interface validated through an automated 20-test Pytest suite.", style_body_noindent))

    # -------------------------------------------------------------
    # SECTION II: LITERATURE REVIEW
    # -------------------------------------------------------------
    story.append(Paragraph("II. LITERATURE REVIEW", style_sec))

    story.append(Paragraph("A. Predictive Maintenance in Industry 4.0", style_subsec))
    story.append(Paragraph(
        "Predictive maintenance has transitioned from offline vibration spectral analysis toward real-time IoT telemetry analytics [3], [4]. Zonta et al. [4] reviewed over 100 industrial implementations, "
        "finding that sensor-driven condition monitoring reduces unexpected outages by up to 45% and lowers maintenance costs by 25%. However, Dalzochio et al. [5] highlighted that industrial telemetry is frequently "
        "beset by class imbalance, where healthy states represent over 95% of recorded instances. In CNC milling, failure mechanisms are governed by physical cutting interactions: tool flank wear obeys Taylor's tool life "
        "equation, power consumption relates directly to torque and angular velocity, and thermal dissipation governs heat transfer between workpiece and tool [1], [3].",
        style_body
    ))

    story.append(Paragraph("B. Machine Learning Algorithms for PdM", style_subsec))
    story.append(Paragraph(
        "Ensemble learning consistently demonstrates superiority over standard linear and kernel classifiers on industrial tabular datasets [3]. Chen and Guestrin [8] established Extreme Gradient Boosting (XGBoost), "
        "which employs second-order Taylor loss expansions and column subsampling. Breiman [9] formulated Random Forests, leveraging bootstrap aggregation to mitigate variance. Carvalho et al. [3] documented that "
        "tree ensembles achieve superior F1 scores on mechanical telemetry compared to support vector machines and feedforward neural networks, primarily because decision trees naturally capture step-function thresholds "
        "inherent to physical safety boundaries.",
        style_body
    ))

    story.append(Paragraph("C. Class Imbalance Mitigation Strategies", style_subsec))
    story.append(Paragraph(
        "Under severe class imbalance, unweighted empirical risk minimization biases classifiers toward the majority class [10]. Chawla et al. [10] developed SMOTE, synthesizing minority points along k-nearest neighbors. "
        "However, synthetic interpolation in physical thermodynamic spaces risks producing unphysical sensor combinations (e.g., impossible torque-speed combinations). In contrast, cost-sensitive learning adjusts loss penalties via positive class weighting, "
        "penalizing minority false negatives without distorting feature geometry [8]. Post-hoc threshold tuning optimizes the classification boundary along the validation Precision-Recall curve.",
        style_body
    ))

    story.append(Paragraph("D. Explainable AI and Shapley Attributions", style_subsec))
    story.append(Paragraph(
        "Ribeiro et al. [11] proposed LIME, using local surrogate linear approximations. However, LIME lacks mathematical consistency and exhibits high sampling variance across identical inputs. Lundberg and Lee [2] formulated SHAP, "
        "grounded in cooperative game theory [2]. Lundberg et al. [12] introduced TreeSHAP, enabling polynomial-time O(TLD^2) exact computation of Shapley values for decision trees. Steurtewagen and Van den Poel [6] "
        "and Gawde et al. [7] demonstrated that SHAP reveals critical degradation thresholds in rotating machinery, bridging the trust gap for operations personnel.",
        style_body
    ))

    story.append(Paragraph("E. Research Gap and Integration Challenge", style_subsec))
    story.append(Paragraph(
        "While individual studies explore XGBoost, SMOTE, or SHAP in isolation, existing literature rarely integrates leakage-free preprocessing, empirical class-imbalance benchmarking, multi-objective utility model selection, "
        "exact local/global SHAP, deterministic natural-language diagnostics, and interactive what-if simulation into a unified industrial architecture. X-Maintain bridges this integration gap.",
        style_body
    ))

    # TABLE I: RELATED WORK TABLE (Wrapped in KeepTogether)
    t1_title = Paragraph("TABLE I: COMPARATIVE ANALYSIS OF RELATED WORK", style_table_title)
    table1_data = [
        [Paragraph("<b>Ref.</b>", style_ref), Paragraph("<b>Yr</b>", style_ref), Paragraph("<b>Method</b>", style_ref), Paragraph("<b>XAI</b>", style_ref), Paragraph("<b>Focus & Contribution</b>", style_ref), Paragraph("<b>Limitation / Gap</b>", style_ref)],
        [Paragraph("[1] Matzka", style_ref), Paragraph("20", style_ref), Paragraph("RF / kNN", style_ref), Paragraph("SHAP/PDP", style_ref), Paragraph("AI4I benchmark introduction", style_ref), Paragraph("Retained leakage cols", style_ref)],
        [Paragraph("[2] Lundberg", style_ref), Paragraph("17", style_ref), Paragraph("General", style_ref), Paragraph("SHAP", style_ref), Paragraph("Game-theoretic XAI theory", style_ref), Paragraph("Theoretical framework", style_ref)],
        [Paragraph("[3] Carvalho", style_ref), Paragraph("19", style_ref), Paragraph("Survey", style_ref), Paragraph("None", style_ref), Paragraph("Comprehensive ML in PdM", style_ref), Paragraph("No XAI evaluation", style_ref)],
        [Paragraph("[4] Zonta", style_ref), Paragraph("20", style_ref), Paragraph("Survey", style_ref), Paragraph("None", style_ref), Paragraph("Industry 4.0 PdM review", style_ref), Paragraph("Lacks XAI integration", style_ref)],
        [Paragraph("[5] Dalzochio", style_ref), Paragraph("20", style_ref), Paragraph("Survey", style_ref), Paragraph("Rules", style_ref), Paragraph("Reasoning in Industry 4.0", style_ref), Paragraph("No post-hoc ML XAI", style_ref)],
        [Paragraph("[6] Steurtewagen", style_ref), Paragraph("21", style_ref), Paragraph("RF / Logit", style_ref), Paragraph("SHAP", style_ref), Paragraph("Sensor interpretability", style_ref), Paragraph("No sensitivity engine", style_ref)],
        [Paragraph("[7] Gawde", style_ref), Paragraph("21", style_ref), Paragraph("XGBoost", style_ref), Paragraph("SHAP/LIME", style_ref), Paragraph("Rotating machine XAI", style_ref), Paragraph("Lacks deployment UI", style_ref)],
        [Paragraph("[8] Chen", style_ref), Paragraph("16", style_ref), Paragraph("XGBoost", style_ref), Paragraph("Gain", style_ref), Paragraph("Scalable tree boosting", style_ref), Paragraph("Black-box without SHAP", style_ref)],
        [Paragraph("[9] Breiman", style_ref), Paragraph("01", style_ref), Paragraph("RF", style_ref), Paragraph("MDI", style_ref), Paragraph("Bagged ensemble trees", style_ref), Paragraph("Impurity bias in XAI", style_ref)],
        [Paragraph("[10] Chawla", style_ref), Paragraph("02", style_ref), Paragraph("SMOTE", style_ref), Paragraph("None", style_ref), Paragraph("Minority over-sampling", style_ref), Paragraph("Synthetic geometry distortion", style_ref)],
        [Paragraph("[12] Lundberg", style_ref), Paragraph("20", style_ref), Paragraph("TreeSHAP", style_ref), Paragraph("TreeSHAP", style_ref), Paragraph("Polynomial tree XAI", style_ref), Paragraph("Algorithms only", style_ref)],
        [Paragraph("<b>This Work</b>", style_ref), Paragraph("<b>26</b>", style_ref), Paragraph("<b>XGBoost</b>", style_ref), Paragraph("<b>TreeSHAP</b>", style_ref), Paragraph("<b>End-to-end X-Maintain</b>", style_ref), Paragraph("<b>Integrated solution</b>", style_ref)]
    ]
    t1 = Table(table1_data, colWidths=[40, 16, 42, 38, 62, 53])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ('LINEABOVE', (0,0), (-1,0), 0.8, colors.HexColor("#0f172a")),
        ('LINEBELOW', (0,0), (-1,0), 0.8, colors.HexColor("#0f172a")),
        ('LINEBELOW', (0,-1), (-1,-1), 0.8, colors.HexColor("#0f172a")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.2),
        ('LEFTPADDING', (0,0), (-1,-1), 1),
        ('RIGHTPADDING', (0,0), (-1,-1), 1),
    ]))
    story.append(KeepTogether([t1_title, t1, Spacer(1, 4)]))

    # -------------------------------------------------------------
    # SECTION III: DATASET & PROBLEM FORMULATION
    # -------------------------------------------------------------
    story.append(Paragraph("III. DATASET AND PROBLEM FORMULATION", style_sec))

    story.append(Paragraph("A. Dataset Overview and Synthetic Origin", style_subsec))
    story.append(Paragraph(
        "We utilize the AI4I 2020 Predictive Maintenance Dataset [1], hosted by the UCI Machine Learning Repository. "
        "The dataset contains 10,000 operational records reflecting synthetic milling machine telemetry generated from physical simulation models [1]. "
        "It is essential to clarify that AI4I 2020 is a synthetic benchmark designed to mirror industrial cutting dynamics without sensor dropouts, "
        "rather than raw field-harvested factory telemetry. Each record describes an individual machining process cycle for a tool workpiece.",
        style_body
    ))

    story.append(Paragraph("B. Physical Failure Modes and Underlying Equations", style_subsec))
    story.append(Paragraph(
        "The simulation model encodes five distinct physical failure mechanisms governing the milling operation [1]:",
        style_body
    ))
    story.append(Paragraph(
        "1) <i>Tool Wear Failure (TWF):</i> Cutter contact friction gradually wears down the milling tool. Based on Taylor's tool life model, "
        "failure occurs when cumulative contact duration satisfies <i>Tool wear</i> &ge; [200, 240] minutes, depending on tool variant quality.",
        style_body_noindent
    ))
    story.append(Paragraph(
        "2) <i>Heat Dissipation Failure (HDF):</i> Heat generated during cutting must be dissipated into the surrounding atmosphere. "
        "When the temperature difference &Delta;<i>T</i> = <i>Process temperature</i> &minus; <i>Air temperature</i> &lt; 8.6 K while rotational speed "
        "&omega; &lt; 1,380 rpm, cooling airflow is insufficient, triggering thermal seizure.",
        style_body_noindent
    ))
    story.append(Paragraph(
        "3) <i>Power Failure (PWF):</i> The mechanical power required for machining is defined by the rotational equation <i>P</i> = &tau; &times; (2&pi;&omega;/60), "
        "where &tau; is torque in Nm and &omega; is rotational speed in rpm. Power failure occurs when <i>P</i> &lt; 3,500 W (stalling) or <i>P</i> &gt; 9,000 W (overload).",
        style_body_noindent
    ))
    story.append(Paragraph(
        "4) <i>Overstrain Failure (OSF):</i> The product of tool wear and torque reflects mechanical stress intensity. When &tau; &times; [<i>Tool wear</i>] exceeds "
        "variant thresholds (11,000 min&middot;Nm for Type L, 12,000 for Type M, and 13,000 for Type H), the tool fractures.",
        style_body_noindent
    ))
    story.append(Paragraph(
        "5) <i>Random Failure (RNF):</i> Unforeseen mechanical defects occur stochastically with a 0.1% background failure probability.",
        style_body_noindent
    ))

    story.append(Paragraph("C. Target Variable and Severe Class Imbalance", style_subsec))
    story.append(Paragraph(
        "The primary modeling objective is binary failure prediction: <i>Machine failure</i> &isin; {0, 1}. "
        "Across the 10,000 instances, exactly 9,661 samples (96.61%) represent nominal operation (Class 0), while 339 samples (3.39%) represent failure events (Class 1). "
        "This creates an extreme class imbalance ratio of 28.50 : 1. Fig. 1 illustrates the stark target distribution. "
        "Optimizing naive overall accuracy under this distribution would yield a non-functional model: predicting 'no failure' universally yields 96.61% accuracy while capturing 0% of breakdowns.",
        style_body
    ))

    img_target = os.path.join(BASE_DIR, 'artifacts', 'figures', 'target_distribution.png')
    if os.path.exists(img_target):
        story.append(Image(img_target, width=238, height=147))
        story.append(Paragraph("Fig. 1. Class imbalance distribution showing 28.5:1 ratio between nominal operation and failure.", style_caption))

    story.append(Paragraph("D. Target Leakage Prevention", style_subsec))
    story.append(Paragraph(
        "The raw dataset records the five failure-mode indicators (TWF, HDF, PWF, OSF, RNF). "
        "Crucially, the target variable is a logical union of these modes: <i>Machine failure</i> = TWF &or; HDF &or; PWF &or; OSF &or; RNF. "
        "Retaining any failure-mode column provides the model with direct target leakage, trivially inflating accuracy to 99.9% while rendering the model useless for proactive forecasting. "
        "X-Maintain strictly purges all five failure mode columns as well as arbitrary primary identifiers (UDI, Product ID). This forces the algorithms to learn "
        "the underlying physical thermodynamic boundaries purely from non-leaking raw telemetry.",
        style_body
    ))

    # TABLE II: DATASET FEATURES (KeepTogether)
    t2_title = Paragraph("TABLE II: FINAL MODELING FEATURES", style_table_title)
    table2_data = [
        [Paragraph("<b>Feature Name</b>", style_ref), Paragraph("<b>Type</b>", style_ref), Paragraph("<b>Range / Values</b>", style_ref), Paragraph("<b>Physical Role</b>", style_ref)],
        [Paragraph("Air temperature (K)", style_ref), Paragraph("Float", style_ref), Paragraph("295.3 - 304.5 K", style_ref), Paragraph("Ambient factory temp", style_ref)],
        [Paragraph("Process temperature (K)", style_ref), Paragraph("Float", style_ref), Paragraph("305.7 - 313.8 K", style_ref), Paragraph("Workpiece contact temp", style_ref)],
        [Paragraph("Rotational speed (rpm)", style_ref), Paragraph("Integer", style_ref), Paragraph("1,168 - 2,886 rpm", style_ref), Paragraph("Spindle rotational velocity", style_ref)],
        [Paragraph("Torque (Nm)", style_ref), Paragraph("Float", style_ref), Paragraph("3.8 - 76.6 Nm", style_ref), Paragraph("Mechanical cutting torque", style_ref)],
        [Paragraph("Tool wear (min)", style_ref), Paragraph("Integer", style_ref), Paragraph("0 - 253 min", style_ref), Paragraph("Accumulated tool contact", style_ref)],
        [Paragraph("Type_H", style_ref), Paragraph("Binary", style_ref), Paragraph("{0, 1} (20%)", style_ref), Paragraph("High quality variant", style_ref)],
        [Paragraph("Type_L", style_ref), Paragraph("Binary", style_ref), Paragraph("{0, 1} (50%)", style_ref), Paragraph("Low quality variant", style_ref)],
        [Paragraph("Type_M", style_ref), Paragraph("Binary", style_ref), Paragraph("{0, 1} (30%)", style_ref), Paragraph("Medium quality variant", style_ref)]
    ]
    t2 = Table(table2_data, colWidths=[75, 35, 65, 76])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ('LINEABOVE', (0,0), (-1,0), 0.8, colors.HexColor("#0f172a")),
        ('LINEBELOW', (0,0), (-1,0), 0.8, colors.HexColor("#0f172a")),
        ('LINEBELOW', (0,-1), (-1,-1), 0.8, colors.HexColor("#0f172a")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.2),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(KeepTogether([t2_title, t2, Spacer(1, 4)]))

    story.append(Paragraph("E. Exploratory Data Analysis and Telemetry Correlations", style_subsec))
    story.append(Paragraph(
        "Exploratory analysis reveals key operational relationships. Fig. 2 displays numerical sensor histograms, showing normal distributions for temperatures and torque, "
        "and a right-skewed distribution for spindle speed. Fig. 3 shows the feature correlation matrix. Notice the strong negative correlation (r = -0.88) between rotational speed and torque, "
        "consistent with constant-power cutting regimes. Tool wear exhibits near-zero linear correlation with other variables, signifying independent abrasive accumulation.",
        style_body
    ))

    img_hist = os.path.join(BASE_DIR, 'artifacts', 'figures', 'numerical_distributions.png')
    if os.path.exists(img_hist):
        story.append(Image(img_hist, width=242, height=164))
        story.append(Paragraph("Fig. 2. Empirical distributions of the continuous industrial sensor features.", style_caption))

    img_corr = os.path.join(BASE_DIR, 'artifacts', 'figures', 'correlation_heatmap.png')
    if os.path.exists(img_corr):
        story.append(Image(img_corr, width=238, height=190))
        story.append(Paragraph("Fig. 3. Pearson correlation matrix highlighting the strong Torque-RPM physical dependency (r = -0.88).", style_caption))

    img_fail_type = os.path.join(BASE_DIR, 'artifacts', 'figures', 'failure_by_type.png')
    if os.path.exists(img_fail_type):
        story.append(Image(img_fail_type, width=238, height=147))
        story.append(Paragraph("Fig. 4. Empirical machine failure rate categorized by product variant quality (Type L, M, H).", style_caption))

    story.append(Paragraph(
        "Fig. 4 highlights failure propensity across product variants: Type L (low quality) exhibits the highest failure rate (~3.9%), while Type H exhibits the lowest (~2.1%). "
        "This confirms that material quality directly alters wear tolerance and fracture thresholds in milling operations.",
        style_body
    ))

    img_feat_fail = os.path.join(BASE_DIR, 'artifacts', 'figures', 'feature_by_failure.png')
    if os.path.exists(img_feat_fail):
        story.append(Image(img_feat_fail, width=242, height=164))
        story.append(Paragraph("Fig. 5. Overlapping sensor distributions stratified by nominal operation vs. machine breakdown.", style_caption))

    story.append(Paragraph(
        "Fig. 5 compares feature distributions conditioned on failure status. While nominal machines cluster around 40 Nm torque and moderate tool wear, "
        "failure instances exhibit severe bimodal extremes: clusters of high torque (>55 Nm), elevated tool wear (>200 min), and localized low rotational velocities (<1,380 rpm).",
        style_body
    ))

    # -------------------------------------------------------------
    # SECTION IV: PROPOSED X-MAINTAIN FRAMEWORK
    # -------------------------------------------------------------
    story.append(Paragraph("IV. PROPOSED X-MAINTAIN FRAMEWORK", style_sec))
    story.append(Paragraph(
        "The overall architectural pipeline of X-Maintain is illustrated in Fig. 6, comprising five modular layers: "
        "1) Ingestion & Leakage Purge, 2) Stratified Preprocessing Pipeline, 3) Cost-Sensitive Multi-Model Training, "
        "4) Game-Theoretic TreeSHAP Attribution, and 5) Interactive Streamlit Deployment with What-If Sensitivity Simulation.",
        style_body
    ))

    img_arch = os.path.join(BASE_DIR, 'artifacts', 'figures', 'architecture_diagram.png')
    if os.path.exists(img_arch):
        story.append(Image(img_arch, width=244, height=138))
        story.append(Paragraph("Fig. 6. Overall architecture of the proposed X-Maintain framework.", style_caption))

    img_wf = os.path.join(BASE_DIR, 'artifacts', 'figures', 'workflow_diagram.png')
    if os.path.exists(img_wf):
        story.append(Image(img_wf, width=244, height=122))
        story.append(Paragraph("Fig. 7. End-to-end dataflow and diagnostic inference sequence.", style_caption))

    story.append(Paragraph("A. Preprocessing and Feature Engineering", style_subsec))
    story.append(Paragraph(
        "Data partitioning enforces stratified sampling with a fixed seed (RANDOM_STATE = 42): 70% Train (6,999 records, 237 failures), "
        "10% Validation (1,000 records, 34 failures), and 20% Test (2,001 records, 68 failures). "
        "To strictly avoid data snooping, StandardScaler is fitted exclusively on training numerical features and applied to validation/test partitions. "
        "The categorical Type variable is one-hot encoded into three binary indicators: Type_H, Type_L, and Type_M. "
        "Feature names are sanitized from raw brackets to parentheses for XGBoost compatibility.",
        style_body
    ))

    story.append(Paragraph("B. Model Selection Criterion", style_subsec))
    story.append(Paragraph(
        "Model selection is formulated via a weighted utility function balancing failure capture against operational precision: "
        "Utility = 0.4 &times; Recall + 0.3 &times; F1 + 0.3 &times; ROC-AUC. "
        "This utility explicitly rewards high sensitivity on catastrophic failures while penalizing false alarm floods.",
        style_body
    ))

    # -------------------------------------------------------------
    # SECTION V: MACHINE LEARNING MODELS & IMBALANCE HANDLING
    # -------------------------------------------------------------
    story.append(Paragraph("V. MACHINE LEARNING MODELS & THEORETICAL FOUNDATIONS", style_sec))

    story.append(Paragraph("A. Baseline: Cost-Weighted Logistic Regression", style_subsec))
    story.append(Paragraph(
        "We formulate an L2-regularized linear baseline with balanced class weights w_k = N / (2 N_k). "
        "The class-weighted cross-entropy loss function is defined as:",
        style_body
    ))
    story.append(Paragraph("L_CW(&theta;) = &minus;&Sigma; [ w_1 y_i log(y_hat_i) + w_0 (1 &minus; y_i) log(1 &minus; y_hat_i) ] + &lambda; ||&theta;||_2^2", style_equation))
    story.append(Paragraph(
        "where y_hat_i = &sigma;(&theta;^T x_i) = 1 / (1 + exp(&minus;&theta;^T x_i)). Logistic Regression establishes the baseline performance reachable "
        "under linear hyperplanes.",
        style_body
    ))

    story.append(Paragraph("B. Bagging Ensemble: Random Forest", style_subsec))
    story.append(Paragraph(
        "Random Forest constructs an ensemble of B = 200 de-correlated decision trees with balanced bootstrap sampling [9]. "
        "At each split node, a random subset of features m = &radic;p is evaluated using Gini impurity reduction: "
        "&Delta;I(S, A) = I(S) &minus; &Sigma; (|S_v| / |S|) I(S_v). Predictions represent ensemble majority voting: H(x) = (1/B) &Sigma; h_b(x).",
        style_body
    ))

    story.append(Paragraph("C. Gradient Boosting: Extreme Gradient Boosting (XGBoost)", style_subsec))
    story.append(Paragraph(
        "XGBoost constructs additive regression trees minimizing regularized objective via second-order Taylor expansion [8]:",
        style_body
    ))
    story.append(Paragraph("L^(t) &approx; &Sigma; [ g_i f_t(x_i) + 0.5 h_i f_t^2(x_i) ] + &gamma; T + 0.5 &lambda; &Sigma; w_j^2", style_equation))
    story.append(Paragraph(
        "where g_i = &part; l(y_i, y_hat^(t-1)) / &part; y_hat^(t-1) and h_i = &part;^2 l(y_i, y_hat^(t-1)) / &part; (y_hat^(t-1))^2 are first and second order gradients. "
        "The optimal leaf weight w_j^* and split gain are analytically computed as:",
        style_body
    ))
    story.append(Paragraph("w_j^* = &minus; &Sigma; g_i / (&Sigma; h_i + &lambda;), &nbsp;&nbsp; Gain = 0.5 [ G_L^2/(H_L+&lambda;) + G_R^2/(H_R+&lambda;) &minus; (G_L+G_R)^2/(H_L+H_R+&lambda;) ] &minus; &gamma;", style_equation))
    story.append(Paragraph(
        "To counteract the 28.5:1 class disparity, dynamic cost-sensitive weighting is configured via: "
        "scale_pos_weight = N_negative / N_positive = 6,762 / 237 = 28.53. "
        "This forces gradient steps to penalize false negatives 28.53 times more heavily than false positives.",
        style_body
    ))

    story.append(Paragraph("D. TreeSHAP Attribution Formulation", style_subsec))
    story.append(Paragraph(
        "Shapley values allocate additive feature contributions relative to base expected output [2], [12]:",
        style_body
    ))
    story.append(Paragraph("&phi;_i(x) = &Sigma;_{S &sube; F \\ {i}} [ |S|! (|F| &minus; |S| &minus; 1)! / |F|! ] [ f_x(S &cup; {i}) &minus; f_x(S) ]", style_equation))
    story.append(Paragraph(
        "For tree ensembles, TreeSHAP computes exact values in polynomial time O(TLD^2) (where T is number of trees, L is max leaves, and D is max depth) "
        "by recursively tracking leaf weights across split paths, satisfying efficiency, symmetry, dummy, and additivity axioms [12].",
        style_body
    ))

    # -------------------------------------------------------------
    # SECTION VI: EXPERIMENTAL SETUP
    # -------------------------------------------------------------
    story.append(Paragraph("VI. EXPERIMENTAL SETUP", style_sec))
    story.append(Paragraph(
        "Experiments are executed on Python 3.9.13, Scikit-learn 1.6.1, XGBoost 2.1.4, SHAP 0.49.1, and Imbalanced-Learn 0.12.4. "
        "Evaluation metrics include Accuracy = (TP+TN)/(P+N), Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2&times;P&times;R/(P+R), "
        "Receiver Operating Characteristic Area Under Curve (ROC-AUC), and Precision-Recall Area Under Curve (PR-AUC = &Sigma; (R_k - R_{k-1}) P_k). "
        "PR-AUC is vital for imbalanced failure forecasting because it ignores the overwhelming true negative count, focusing exclusively on minority resolution.",
        style_body
    ))

    # -------------------------------------------------------------
    # SECTION VII: RESULTS AND DISCUSSION
    # -------------------------------------------------------------
    story.append(Paragraph("VII. RESULTS AND DISCUSSION", style_sec))

    story.append(Paragraph("A. Primary Model Benchmark", style_subsec))
    story.append(Paragraph(
        "Table III documents the empirical evaluation across all 2,001 held-out test cases (68 true machine breakdowns). "
        "XGBoost achieves the top selection score (0.8064), balancing 73.53% Recall with 73.53% Precision and 0.9724 ROC-AUC.",
        style_body
    ))

    # TABLE III: PRIMARY MODEL BENCHMARK (KeepTogether)
    t3_title = Paragraph("TABLE III: PERFORMANCE BENCHMARK (TEST SPLIT, N = 2,001)", style_table_title)
    t3_data = [
        [Paragraph("<b>Model</b>", style_ref), Paragraph("<b>Acc.</b>", style_ref), Paragraph("<b>Prec.</b>", style_ref), Paragraph("<b>Recall</b>", style_ref), Paragraph("<b>F1</b>", style_ref), Paragraph("<b>ROC</b>", style_ref), Paragraph("<b>PR</b>", style_ref), Paragraph("<b>Score</b>", style_ref)],
        [Paragraph("Logistic Regression", style_ref), Paragraph("83.61%", style_ref), Paragraph("14.67%", style_ref), Paragraph("<b>79.41%</b>", style_ref), Paragraph("24.77%", style_ref), Paragraph("0.8949", style_ref), Paragraph("0.4187", style_ref), Paragraph("0.6604", style_ref)],
        [Paragraph("Random Forest", style_ref), Paragraph("97.75%", style_ref), Paragraph("<b>87.10%</b>", style_ref), Paragraph("39.71%", style_ref), Paragraph("54.55%", style_ref), Paragraph("0.9556", style_ref), Paragraph("0.6930", style_ref), Paragraph("0.6091", style_ref)],
        [Paragraph("<b>XGBoost (Selected)</b>", style_ref), Paragraph("<b>98.20%</b>", style_ref), Paragraph("73.53%", style_ref), Paragraph("73.53%", style_ref), Paragraph("<b>73.53%</b>", style_ref), Paragraph("<b>0.9724</b>", style_ref), Paragraph("<b>0.7934</b>", style_ref), Paragraph("<b>0.8064</b>", style_ref)]
    ]
    t3 = Table(t3_data, colWidths=[65, 27, 27, 27, 27, 27, 27, 26])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ('LINEABOVE', (0,0), (-1,0), 0.8, colors.HexColor("#0f172a")),
        ('LINEBELOW', (0,0), (-1,0), 0.8, colors.HexColor("#0f172a")),
        ('LINEBELOW', (0,-1), (-1,-1), 0.8, colors.HexColor("#0f172a")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.2),
        ('LEFTPADDING', (0,0), (-1,-1), 1),
        ('RIGHTPADDING', (0,0), (-1,-1), 1),
    ]))
    story.append(KeepTogether([t3_title, t3, Spacer(1, 4)]))

    story.append(Paragraph("B. Analysis of Precision-Recall Trade-offs", style_subsec))
    story.append(Paragraph(
        "The benchmark reveals three distinct operating trade-offs: "
        "1) <i>Logistic Regression</i> attains the highest raw recall (79.41%, capturing 54/68 failures), but precision collapses to 14.67%, generating 314 false alarms that cause alert fatigue. "
        "2) <i>Random Forest</i> displays 87.10% precision, but misses over 60% of true failure events (Recall = 39.71%, 41 missed failures), which is unacceptable in plant operations. "
        "3) <i>XGBoost</i> provides the optimal trade-off: 50 of 68 breakdowns captured (73.53% Recall) with only 18 false alarms (73.53% Precision).",
        style_body
    ))

    # TABLE IV: IMBALANCE EXPERIMENTS (KeepTogether)
    t4_title = Paragraph("TABLE IV: CLASS IMBALANCE MITIGATION STUDY", style_table_title)
    t4_data = [
        [Paragraph("<b>Experiment Config</b>", style_ref), Paragraph("<b>Acc.</b>", style_ref), Paragraph("<b>Prec.</b>", style_ref), Paragraph("<b>Recall</b>", style_ref), Paragraph("<b>F1</b>", style_ref), Paragraph("<b>ROC</b>", style_ref), Paragraph("<b>PR</b>", style_ref), Paragraph("<b>Score</b>", style_ref)],
        [Paragraph("Exp 4b: XGBoost + SMOTE", style_ref), Paragraph("97.85%", style_ref), Paragraph("65.06%", style_ref), Paragraph("<b>79.41%</b>", style_ref), Paragraph("71.52%", style_ref), Paragraph("0.9620", style_ref), Paragraph("<b>0.7996</b>", style_ref), Paragraph("<b>0.8208</b>", style_ref)],
        [Paragraph("Exp 4c: XGBoost + Thresh (&tau;*=0.61)", style_ref), Paragraph("<b>98.35%</b>", style_ref), Paragraph("76.92%", style_ref), Paragraph("73.53%", style_ref), Paragraph("<b>75.19%</b>", style_ref), Paragraph("<b>0.9724</b>", style_ref), Paragraph("0.7934", style_ref), Paragraph("0.8114", style_ref)],
        [Paragraph("Exp 4a: XGBoost + Cost-Weight", style_ref), Paragraph("98.20%", style_ref), Paragraph("73.53%", style_ref), Paragraph("73.53%", style_ref), Paragraph("73.53%", style_ref), Paragraph("<b>0.9724</b>", style_ref), Paragraph("0.7934", style_ref), Paragraph("0.8064", style_ref)],
        [Paragraph("Exp 3: XGBoost Unweighted", style_ref), Paragraph("98.30%", style_ref), Paragraph("<b>80.36%</b>", style_ref), Paragraph("66.18%", style_ref), Paragraph("72.58%", style_ref), Paragraph("0.9653", style_ref), Paragraph("0.7874", style_ref), Paragraph("0.7720", style_ref)],
        [Paragraph("Exp 1: Logistic Baseline", style_ref), Paragraph("83.61%", style_ref), Paragraph("14.67%", style_ref), Paragraph("<b>79.41%</b>", style_ref), Paragraph("24.77%", style_ref), Paragraph("0.8949", style_ref), Paragraph("0.4187", style_ref), Paragraph("0.6604", style_ref)],
        [Paragraph("Exp 2: Random Forest", style_ref), Paragraph("97.75%", style_ref), Paragraph("87.10%", style_ref), Paragraph("39.71%", style_ref), Paragraph("54.55%", style_ref), Paragraph("0.9556", style_ref), Paragraph("0.6930", style_ref), Paragraph("0.6091", style_ref)]
    ]
    t4 = Table(t4_data, colWidths=[80, 24, 25, 25, 25, 25, 25, 24])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ('LINEABOVE', (0,0), (-1,0), 0.8, colors.HexColor("#0f172a")),
        ('LINEBELOW', (0,0), (-1,0), 0.8, colors.HexColor("#0f172a")),
        ('LINEBELOW', (0,-1), (-1,-1), 0.8, colors.HexColor("#0f172a")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.2),
        ('LEFTPADDING', (0,0), (-1,-1), 1),
        ('RIGHTPADDING', (0,0), (-1,-1), 1),
    ]))
    story.append(KeepTogether([t4_title, t4, Spacer(1, 4)]))

    story.append(Paragraph("C. Systematic Imbalance Experiments Comparison", style_subsec))
    story.append(Paragraph(
        "Fig. 8 illustrates the systematic comparison across all six experimental imbalance configurations. "
        "SMOTE (Exp 4b) attains the highest failure recall (79.41%), but sacrifices precision (65.06%) due to artificial interpolation across continuous thermodynamic boundaries. "
        "Cost-sensitive XGBoost (Exp 4a) preserves original sensor geometries while delivering balanced 73.53% precision and recall. "
        "Threshold tuning (Exp 4c, &tau;* = 0.6123) pushes overall test accuracy to 98.35% and F1 to 75.19%.",
        style_body
    ))

    img_imb_comp = os.path.join(BASE_DIR, 'artifacts', 'figures', 'imbalance_experiments_comparison.png')
    if os.path.exists(img_imb_comp):
        story.append(Image(img_imb_comp, width=242, height=121))
        story.append(Paragraph("Fig. 8. Comparison of precision, recall, F1, and utility across all imbalance configurations.", style_caption))

    story.append(Paragraph("D. Visual Discrimination Performance", style_subsec))
    story.append(Paragraph(
        "Fig. 9, 10, and 11 display the visual performance curves. "
        "Fig. 9 presents the confusion matrix for cost-sensitive XGBoost on the test split (N = 2,001), showing 1,915 true negatives, 50 true positives, 18 false positives, and 18 false negatives. "
        "Fig. 10 shows the multi-model ROC curves where XGBoost leads with AUC = 0.9724. "
        "Fig. 11 shows the Precision-Recall curves where XGBoost dominates with PR-AUC = 0.7934.",
        style_body
    ))

    img_cm = os.path.join(BASE_DIR, 'artifacts', 'figures', 'cm_xgboost.png')
    if os.path.exists(img_cm):
        story.append(Image(img_cm, width=216, height=176))
        story.append(Paragraph("Fig. 9. Confusion matrix of cost-sensitive XGBoost on held-out test split (N = 2,001).", style_caption))

    img_cm_lr = os.path.join(BASE_DIR, 'artifacts', 'figures', 'cm_logistic_regression.png')
    if os.path.exists(img_cm_lr):
        story.append(Image(img_cm_lr, width=216, height=176))
        story.append(Paragraph("Fig. 10. Baseline Logistic Regression confusion matrix exhibiting severe false alarm flooding (314 FPs).", style_caption))

    img_cm_rf = os.path.join(BASE_DIR, 'artifacts', 'figures', 'cm_random_forest.png')
    if os.path.exists(img_cm_rf):
        story.append(Image(img_cm_rf, width=216, height=176))
        story.append(Paragraph("Fig. 11. Random Forest confusion matrix exhibiting dangerous missed breakdowns (41 FNs, Recall = 39.7%).", style_caption))

    img_roc = os.path.join(BASE_DIR, 'artifacts', 'figures', 'roc_curves.png')
    if os.path.exists(img_roc):
        story.append(Image(img_roc, width=234, height=175))
        story.append(Paragraph("Fig. 12. Multi-model ROC curves demonstrating XGBoost discriminative dominance (AUC = 0.972).", style_caption))

    img_pr = os.path.join(BASE_DIR, 'artifacts', 'figures', 'pr_curves.png')
    if os.path.exists(img_pr):
        story.append(Image(img_pr, width=234, height=175))
        story.append(Paragraph("Fig. 13. Precision-Recall curves on imbalanced failure class (XGBoost PR-AUC = 0.793).", style_caption))

    story.append(Paragraph("E. SHAP Global Attribution", style_subsec))
    story.append(Paragraph(
        "Global feature importance was computed using TreeExplainer across test samples. Table V summarizes the exact mean absolute SHAP values. "
        "Torque (3.74) and Tool Wear (3.06) are the primary drivers of model failure predictions, followed by Rotational Speed (2.07) and Air Temperature (2.01). "
        "The beeswarm distribution in Fig. 15 reveals that high torque and elevated tool wear shift log-odds strongly toward failure, whereas high rotational speeds act as stabilizers under moderate torque.",
        style_body
    ))

    # TABLE V: GLOBAL FEATURE IMPORTANCE (KeepTogether)
    t5_title = Paragraph("TABLE V: GLOBAL FEATURE IMPORTANCE (TREESHAP)", style_table_title)
    t5_data = [
        [Paragraph("<b>Rank</b>", style_ref), Paragraph("<b>Feature Name</b>", style_ref), Paragraph("<b>Mean |SHAP|</b>", style_ref), Paragraph("<b>Physical Operational Role</b>", style_ref)],
        [Paragraph("1", style_ref), Paragraph("Torque (Nm)", style_ref), Paragraph("3.7397", style_ref), Paragraph("Primary driver of rotational overstrain", style_ref)],
        [Paragraph("2", style_ref), Paragraph("Tool wear (min)", style_ref), Paragraph("3.0570", style_ref), Paragraph("Cumulative cutting head abrasion", style_ref)],
        [Paragraph("3", style_ref), Paragraph("Rotational speed (rpm)", style_ref), Paragraph("2.0700", style_ref), Paragraph("Spindle velocity / thermal dissipation", style_ref)],
        [Paragraph("4", style_ref), Paragraph("Air temperature (K)", style_ref), Paragraph("2.0052", style_ref), Paragraph("Ambient environmental boundary temp", style_ref)],
        [Paragraph("5", style_ref), Paragraph("Process temp (K)", style_ref), Paragraph("0.9878", style_ref), Paragraph("Workpiece contact zone temp", style_ref)],
        [Paragraph("6", style_ref), Paragraph("Type_L", style_ref), Paragraph("0.1962", style_ref), Paragraph("Low-quality variant failure propensity", style_ref)],
        [Paragraph("7", style_ref), Paragraph("Type_M", style_ref), Paragraph("0.1736", style_ref), Paragraph("Medium-tier baseline quality variant", style_ref)],
        [Paragraph("8", style_ref), Paragraph("Type_H", style_ref), Paragraph("0.0882", style_ref), Paragraph("High-durability product variant", style_ref)]
    ]
    t5 = Table(t5_data, colWidths=[24, 78, 48, 103])
    t5.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ('LINEABOVE', (0,0), (-1,0), 0.8, colors.HexColor("#0f172a")),
        ('LINEBELOW', (0,0), (-1,0), 0.8, colors.HexColor("#0f172a")),
        ('LINEBELOW', (0,-1), (-1,-1), 0.8, colors.HexColor("#0f172a")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.2),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(KeepTogether([t5_title, t5, Spacer(1, 4)]))

    img_bar = os.path.join(BASE_DIR, 'artifacts', 'figures', 'shap_global_importance.png')
    if os.path.exists(img_bar):
        story.append(Image(img_bar, width=238, height=160))
        story.append(Paragraph("Fig. 14. Mean absolute SHAP values illustrating global feature importance.", style_caption))

    img_bee = os.path.join(BASE_DIR, 'artifacts', 'figures', 'shap_beeswarm.png')
    if os.path.exists(img_bee):
        story.append(Image(img_bee, width=238, height=140))
        story.append(Paragraph("Fig. 15. SHAP beeswarm summary plot showing feature value spectrum vs. failure risk.", style_caption))

    story.append(Paragraph("F. SHAP Dependence and Feature Interactions", style_subsec))
    story.append(Paragraph(
        "TreeSHAP dependence plots uncover nonlinear physical inflection boundaries. "
        "In Fig. 16 (Torque), SHAP value increases sharply above 50 Nm, demonstrating the onset of rotational overstrain. "
        "In Fig. 17 (Tool Wear), SHAP values transition steeply into positive failure risk at approximately 200 minutes, directly mirroring Taylor's cutting tool wear life limit. "
        "In Fig. 18 (Rotational Speed), low speeds below 1,380 rpm coupled with high torque dramatically elevate failure risk due to inadequate cooling airflow.",
        style_body
    ))

    img_dep_tor = os.path.join(BASE_DIR, 'artifacts', 'figures', 'shap_dependence_Torque_Nm.png')
    if os.path.exists(img_dep_tor):
        story.append(Image(img_dep_tor, width=238, height=162))
        story.append(Paragraph("Fig. 16. SHAP dependence for Torque, revealing non-linear risk escalation above 50 Nm.", style_caption))

    img_dep_tool = os.path.join(BASE_DIR, 'artifacts', 'figures', 'shap_dependence_Tool_wear_min.png')
    if os.path.exists(img_dep_tool):
        story.append(Image(img_dep_tool, width=238, height=162))
        story.append(Paragraph("Fig. 17. SHAP dependence for Tool wear, exhibiting inflection at 200 min cutter degradation limit.", style_caption))

    img_dep_rpm = os.path.join(BASE_DIR, 'artifacts', 'figures', 'shap_dependence_Rotational_speed_rpm.png')
    if os.path.exists(img_dep_rpm):
        story.append(Image(img_dep_rpm, width=238, height=162))
        story.append(Paragraph("Fig. 18. SHAP dependence for Rotational speed, showing cooling airflow failure boundary under 1,380 rpm.", style_caption))

    story.append(Paragraph("G. Local Attribution and Case Study", style_subsec))
    story.append(Paragraph(
        "For test instance #11 (True Label = 1, Predicted Probability = 1.000), TreeSHAP decomposes the log-odds output from base expectation E[f(x)] = -3.759 to final prediction f(x) = +11.49 (Fig. 19). "
        "Torque (+7.37) and Tool Wear (+4.19) are the dominant positive contributors pushing the prediction to failure, while Air Temperature (-1.31) provides negative stabilization. "
        "This exact decomposition provides the operator with an indisputable audit trail.",
        style_body
    ))

    img_wf_local = os.path.join(BASE_DIR, 'artifacts', 'figures', 'shap_waterfall.png')
    if os.path.exists(img_wf_local):
        story.append(Image(img_wf_local, width=242, height=160))
        story.append(Paragraph("Fig. 19. Local TreeSHAP waterfall decomposition for test breakdown instance #11.", style_caption))

    story.append(Paragraph("H. Model-Based What-If Sensitivity Analysis", style_subsec))
    story.append(Paragraph(
        "To enable counterfactual parameter exploration, the sensitivity engine perturbs operational variables across the model's learned response surface. "
        "In a verified case study: a high-stress baseline machine (Torque = 65.0 Nm, Tool wear = 210 min, Air temp = 301.5 K, Process temp = 310.8 K, RPM = 1400, Type = L) yields a predicted failure probability of 100.0% (High Risk). "
        "When an operator simulates counterfactual cutter replacement (Tool wear reduced to 30 min) and torque attenuation (reduced to 38.0 Nm), the predicted probability drops to 0.0% (Low Risk), generating a net delta of -100.0 percentage points. "
        "We explicitly caution operators that this represents model sensitivity analysis, not physical causal intervention.",
        style_body
    ))

    # -------------------------------------------------------------
    # SECTION VIII: STREAMLIT SYSTEM IMPLEMENTATION
    # -------------------------------------------------------------
    story.append(Paragraph("VIII. STREAMLIT SYSTEM IMPLEMENTATION", style_sec))
    story.append(Paragraph(
        "The X-Maintain monitoring platform is implemented in Streamlit 1.50.0, featuring a dark industrial control-room aesthetic. "
        "The application is organized into seven operational views: "
        "1) <i>Overview Dashboard:</i> Fleet telemetry KPIs, empirical failure risk zones, and global importance bars. "
        "2) <i>Machine Prediction Console:</i> Manual parameter entry sliders with demonstration presets (Nominal, Wear Failure, Thermal Risk) and dynamic risk alert banners. "
        "3) <i>Explainability Center:</i> High-contrast local waterfall decompositions, top contributing factor cards, and interactive dependence plots. "
        "4) <i>What-If Sensitivity Simulator:</i> Dual-scenario slider comparisons with probability shift delta metrics. "
        "5) <i>Model Performance Analytics:</i> Full benchmark comparison tables, confusion matrices, and ROC/PR curves. "
        "6) <i>Data Insights:</i> Exploratory distributions, correlation heatmaps, and variant failure profiles. "
        "7) <i>About Project:</i> Architecture specifications and limitations.",
        style_body
    ))
    story.append(Paragraph(
        "To guarantee low operational latency in control-room environments, model weights, scalers, and TreeExplainer instances are cached in memory using @st.cache_resource. "
        "Inference and local TreeSHAP attribution execute in under 45 milliseconds on standard CPU hardware, enabling instantaneous what-if slider updates. "
        "Furthermore, deterministic natural-language rules translate numeric Shapley scores into plain-language maintenance recommendations without LLM hallucinations.",
        style_body
    ))

    # -------------------------------------------------------------
    # SECTION IX: SOFTWARE VALIDATION & TESTING
    # -------------------------------------------------------------
    story.append(Paragraph("IX. SOFTWARE VALIDATION AND TESTING", style_sec))
    story.append(Paragraph(
        "To ensure scientific reproducibility and robust execution, the framework includes an automated Pytest suite containing exactly 20 unit tests across three test modules. "
        "Table VI summarizes the verified validation suite, which achieved a 100% pass rate in 3.21 seconds.",
        style_body
    ))

    # TABLE VI: TEST SUITE TABLE (KeepTogether)
    t6_title = Paragraph("TABLE VI: SOFTWARE VALIDATION UNIT TEST SUITE", style_table_title)
    t6_data = [
        [Paragraph("<b>Test Module</b>", style_ref), Paragraph("<b>Target Validation Property</b>", style_ref), Paragraph("<b>Tests</b>", style_ref), Paragraph("<b>Status</b>", style_ref)],
        [Paragraph("test_preprocessing.py", style_ref), Paragraph("Leakage removal, zero nulls, OHE shapes, stratified split preservation", style_ref), Paragraph("8", style_ref), Paragraph("<b>20/20 PASSED</b>", style_ref)],
        [Paragraph("test_prediction.py", style_ref), Paragraph("Model serialization, pipeline transforms, probability bounds [0, 1]", style_ref), Paragraph("5", style_ref), Paragraph("<b>PASSED</b>", style_ref)],
        [Paragraph("test_explainability.py", style_ref), Paragraph("TreeExplainer instantiation, tensor slicing, attribution sorting, NL generation", style_ref), Paragraph("7", style_ref), Paragraph("<b>PASSED</b>", style_ref)],
        [Paragraph("<b>Total Test Suite</b>", style_ref), Paragraph("<b>End-to-end framework verification across all modules</b>", style_ref), Paragraph("<b>20</b>", style_ref), Paragraph("<b>100% PASS</b>", style_ref)]
    ]
    t6 = Table(t6_data, colWidths=[78, 105, 26, 44])
    t6.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ('LINEABOVE', (0,0), (-1,0), 0.8, colors.HexColor("#0f172a")),
        ('LINEBELOW', (0,0), (-1,0), 0.8, colors.HexColor("#0f172a")),
        ('LINEBELOW', (0,-1), (-1,-1), 0.8, colors.HexColor("#0f172a")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.2),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(KeepTogether([t6_title, t6, Spacer(1, 4)]))

    # -------------------------------------------------------------
    # SECTION X: DISCUSSION
    # -------------------------------------------------------------
    story.append(Paragraph("X. DISCUSSION", style_sec))
    story.append(Paragraph(
        "Our empirical results highlight the necessity of balancing failure detection against operational precision. "
        "A model with 95% accuracy that fails to identify tool failure is useless in manufacturing. "
        "Cost-sensitive XGBoost successfully captures 73.53% of failures with an acceptable 73.53% precision rate. "
        "Furthermore, TreeSHAP demystifies the decision surface: plant operators can verify that high torque and accumulated tool wear "
        "are driving risk escalation, transforming raw machine learning scores into actionable maintenance decision support.",
        style_body
    ))
    story.append(Paragraph(
        "A key engineering insight from our SHAP dependence analysis is that machine failure rarely stems from a single isolated sensor anomaly. "
        "Rather, breakdowns emerge from coupled thermodynamic interactions: high torque operating simultaneously with high tool wear creates compound stress, "
        "whereas identical torque under a brand-new tool causes no failure risk. By exposing these joint boundary conditions, TreeSHAP enables maintenance crews "
        "to practice targeted condition-based intervention rather than blanket machine stoppages.",
        style_body
    ))

    # -------------------------------------------------------------
    # SECTION XI: METHODOLOGICAL LIMITATIONS
    # -------------------------------------------------------------
    story.append(Paragraph("XI. METHODOLOGICAL LIMITATIONS", style_sec))
    story.append(Paragraph(
        "We document three distinct methodological limitations of the current study: "
        "1) <i>Synthetic Dataset Nature:</i> The AI4I 2020 dataset simulates milling machine thermodynamics under idealized mathematical models. In live factory environments, sensor telemetry is subject to non-stationary degradation, high-frequency mechanical vibration noise, electromagnetic interference, and variable ambient humidity. "
        "2) <i>Non-Causal Interpretation:</i> Shapley attributions reflect associative model dependencies rather than structural physical causality. Decreasing tool wear in the what-if module reduces predicted model risk, but operational causality requires physical part replacement. "
        "3) <i>Discrete Snapshot Scope:</i> The model operates on tabular snapshots rather than continuous multivariate time-series waveforms.",
        style_body
    ))

    # -------------------------------------------------------------
    # SECTION XII: FUTURE DIRECTIONS
    # -------------------------------------------------------------
    story.append(Paragraph("XII. FUTURE DIRECTIONS", style_sec))
    story.append(Paragraph(
        "Building upon this validated framework, future research will explore three principal frontiers: "
        "1) <i>Temporal Degradation Modeling:</i> Formulating Remaining Useful Life (RUL) regression utilizing Bidirectional LSTMs and Temporal Convolutional Networks (TCNs) applied to raw high-frequency vibration streams. "
        "2) <i>Embedded Edge Deployment:</i> Model compression and 8-bit integer quantization via ONNX Runtime and TensorRT for direct deployment onto low-power industrial edge hardware (e.g., Raspberry Pi 5, NVIDIA Jetson Orin). "
        "3) <i>Causal Machine Learning:</i> Integrating Structural Causal Models (SCMs) and Pearl's do-calculus to transition from observational what-if sensitivity analysis to provably causal prescriptive maintenance recommendations.",
        style_body
    ))

    # -------------------------------------------------------------
    # SECTION XIII: CONCLUSION & FINAL PAGE BALANCING
    # -------------------------------------------------------------
    # Clean page transition to Page 8 so Conclusion and References form a balanced 2-column final page
    story.append(PageBreak())

    story.append(Paragraph("XIII. CONCLUSION", style_sec))
    story.append(Paragraph(
        "This paper presented X-Maintain, an Explainable AI predictive maintenance framework for industrial manufacturing. "
        "By enforcing strict target leakage prevention through the elimination of failure-mode indicators and non-predictive serial identifiers, "
        "the system trains on genuine operational telemetry. Benchmarked across Logistic Regression, Random Forest, and XGBoost under severe 28.5:1 class imbalance, "
        "cost-sensitive XGBoost achieved 98.20% accuracy, 73.53% precision, 73.53% failure recall, 0.9724 ROC-AUC, and 0.7934 PR-AUC on held-out test data. "
        "Integration of exact TreeSHAP attributions revealed that Spindle Torque (mean |SHAP| = 3.74) and Tool Wear (3.06) govern failure risk, "
        "demystifying the black-box boundary. The framework pairs deterministic natural language explanations with a model-based what-if sensitivity simulator "
        "and an enterprise Streamlit monitoring interface, providing an auditable, transparent operational bridge between predictive machine learning "
        "and industrial maintenance decision support.",
        style_body
    ))

    # -------------------------------------------------------------
    # ACKNOWLEDGMENT
    # -------------------------------------------------------------
    story.append(Paragraph("ACKNOWLEDGMENT", style_sec))
    story.append(Paragraph(
        "The author acknowledges the School of Computer Science and Engineering at Vellore Institute of Technology for laboratory infrastructure, "
        "and the creators of the UCI AI4I 2020 Predictive Maintenance Dataset for providing the open-access benchmark.",
        style_body
    ))

    # -------------------------------------------------------------
    # REFERENCES (Balanced across Column 1 and Column 2 of Page 8)
    # -------------------------------------------------------------
    story.append(Paragraph("REFERENCES", style_sec))
    refs = [
        "[1] S. Matzka, \"Explainable Artificial Intelligence for Predictive Maintenance Applications,\" in <i>2020 Third International Conference on Artificial Intelligence for Industries (AI4I)</i>, Irvine, CA, USA, 2020, pp. 69–74, doi: 10.1109/AI4I49448.2020.00023.",
        "[2] S. M. Lundberg and S.-I. Lee, \"A Unified Approach to Interpreting Model Predictions,\" in <i>Advances in Neural Information Processing Systems (NeurIPS 2017)</i>, Long Beach, CA, USA, vol. 30, 2017, pp. 4765–4774.",
        "[3] T. P. Carvalho, F. A. A. Soares, R. Vita, R. d. P. Francisco, J. P. Basto, and S. G. S. Alcalá, \"A systematic literature review of machine learning methods applied to predictive maintenance,\" <i>Computers & Industrial Engineering</i>, vol. 137, p. 106024, Nov. 2019, doi: 10.1016/j.cie.2019.106024.",
        "[4] T. Zonta, C. A. da Costa, R. da Rosa Righi, M. J. de Oliveira, J. Vázquez-Salceda, \"Predictive maintenance in the Industry 4.0: A systematic literature review,\" <i>Computers & Industrial Engineering</i>, vol. 150, p. 106889, Dec. 2020, doi: 10.1016/j.cie.2020.106889.",
        "[5] J. Dalzochio, R. Kunst, E. Pignaton, A. Wiethölter, H. F. Bassani, K. B. C. da Silva, \"Machine learning and reasoning for predictive maintenance in Industry 4.0: Current status and challenges,\" <i>Computers in Industry</i>, vol. 123, p. 103298, Dec. 2020, doi: 10.1016/j.compind.2020.103298.",
        "[6] E. Steurtewagen and D. Van den Poel, \"Adding interpretability to predictive maintenance by machine learning on sensor data,\" <i>Computers & Industrial Engineering</i>, vol. 152, p. 107047, Feb. 2021, doi: 10.1016/j.cie.2020.107047.",
        "[7] P. Gawde, C. Busch, M. Busch, and S. S. Roy, \"Explainable Predictive Maintenance of Rotating Machines Using LIME, SHAP, PDP, ICE,\" in <i>2021 IEEE International Conference on Big Data (Big Data)</i>, Orlando, FL, USA, 2021, pp. 3122–3131, doi: 10.1109/BigData52589.2021.9671607.",
        "[8] T. Chen and C. Guestrin, \"XGBoost: A Scalable Tree Boosting System,\" in <i>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)</i>, San Francisco, CA, USA, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.",
        "[9] L. Breiman, \"Random Forests,\" <i>Machine Learning</i>, vol. 45, no. 1, pp. 5–32, Oct. 2001, doi: 10.1023/A:1010933404324.",
        "[10] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, \"SMOTE: Synthetic Minority Over-sampling Technique,\" <i>Journal of Artificial Intelligence Research</i>, vol. 16, pp. 321–357, Jun. 2002, doi: 10.1613/jair.953.",
        "[11] M. T. Ribeiro, S. Singh, and C. Guestrin, \"\"Why Should I Trust You?\": Explaining the Predictions of Any Classifier,\" in <i>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)</i>, San Francisco, CA, USA, 2016, pp. 1135–1144, doi: 10.1145/2939672.2939778.",
        "[12] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, \"From local explanations to global understanding with explainable AI for trees,\" <i>Nature Machine Intelligence</i>, vol. 2, no. 1, pp. 56–67, Jan. 2020, doi: 10.1038/s42256-019-0138-9."
    ]

    # First 6 references in Column 1
    for r in refs[:6]:
        story.append(Paragraph(r, style_ref))

    # Break to Column 2 of Page 8
    story.append(FrameBreak())

    # Remaining 6 references in Column 2
    for r in refs[6:]:
        story.append(Paragraph(r, style_ref))

    doc.build(story, canvasmaker=IEEENumberedCanvas)
    print(f"PDF generated successfully at {pdf_path}!")

    pdf_doc = fitz.open(pdf_path)
    print(f"Verified PDF Page Count: {len(pdf_doc)} pages")
    return len(pdf_doc)

# =========================================================================
# 2. PYTHON-DOCX GENERATION (Word Document)
# =========================================================================

def build_docx():
    print("Building python-docx Word document...")
    docx_path = os.path.join(BASE_DIR, "X_Maintain_IEEE_Research_Paper.docx")
    doc = docx.Document()

    # Set page margins to 0.5 in (IEEE format)
    for section in doc.sections:
        section.top_margin = Inches(0.5)
        section.bottom_margin = Inches(0.5)
        section.left_margin = Inches(0.5)
        section.right_margin = Inches(0.5)

    # Styles
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title_p.add_run("X-Maintain: An Explainable Machine Learning Framework for Predictive Maintenance and Machine Failure Prediction")
    run_title.font.name = 'Times New Roman'
    run_title.font.size = Pt(18)
    run_title.font.bold = True

    author_p = doc.add_paragraph()
    author_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_author = author_p.add_run("Syagamreddy Gopi Adithya Vardhan Reddy\n")
    run_author.font.name = 'Times New Roman'
    run_author.font.size = Pt(10)
    run_author.font.bold = True
    run_affil = author_p.add_run("Department of Computer Science and Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu 632014, India\n(e-mail: gopi.adithya2022@vitstudent.ac.in)")
    run_affil.font.name = 'Times New Roman'
    run_affil.font.size = Pt(8.5)

    abstract_p = doc.add_paragraph()
    abstract_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run_ab_tag = abstract_p.add_run("Abstract—")
    run_ab_tag.font.name = 'Times New Roman'
    run_ab_tag.font.size = Pt(8.5)
    run_ab_tag.font.bold = True
    run_ab_tag.font.italic = True
    run_ab = abstract_p.add_run(
        "Industrial predictive maintenance (PdM) relies on multivariate sensor streams to anticipate equipment failure before catastrophic downtime occurs. "
        "However, contemporary high-performance machine learning ensembles operate as opaque black boxes, creating an operational trust barrier for factory engineers. "
        "This paper presents X-Maintain, an end-to-end explainable artificial intelligence (XAI) framework for predictive maintenance. Using the benchmark AI4I 2020 Predictive "
        "Maintenance Dataset (10,000 operations), we enforce strict target leakage prevention by excluding failure-mode indicators (TWF, HDF, PWF, OSF, RNF) and primary keys, "
        "yielding an eight-feature operational input matrix. We benchmark Logistic Regression, Random Forest, and XGBoost under severe 28.5:1 class disparity. "
        "Model selection is governed by a domain utility metric (0.4×Recall + 0.3×F1 + 0.3×ROC-AUC). On held-out test split (N = 2,001), cost-sensitive XGBoost achieves 98.20% accuracy, "
        "73.53% precision, 73.53% recall, 0.9724 ROC-AUC, and 0.7934 PR-AUC. Model decisions are explained via TreeSHAP, showing Torque (mean |SHAP| = 3.74) and Tool Wear (3.06) dominate predictions. "
        "Local waterfall attributions, deterministic natural language explanations, what-if sensitivity analysis, and an enterprise Streamlit dashboard complete the validated architecture, backed by 20 passed unit tests."
    )
    run_ab.font.name = 'Times New Roman'
    run_ab.font.size = Pt(8.5)
    run_ab.font.bold = True

    index_p = doc.add_paragraph()
    index_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run_idx_tag = index_p.add_run("Index Terms—")
    run_idx_tag.font.name = 'Times New Roman'
    run_idx_tag.font.size = Pt(8.5)
    run_idx_tag.font.bold = True
    run_idx_tag.font.italic = True
    run_idx = index_p.add_run("Predictive maintenance, Explainable AI (XAI), SHAP, XGBoost, class imbalance, TreeExplainer, industrial IoT, model-based sensitivity analysis.")
    run_idx.font.name = 'Times New Roman'
    run_idx.font.size = Pt(8.5)
    run_idx.font.bold = True

    # Section 2: Two-column layout in Word
    section2 = doc.add_section()
    # Configure 2 columns using XML
    sectPr = section2._sectPr
    cols = sectPr.xpath('./w:cols')
    if cols:
        cols[0].set(qn('w:num'), '2')
        cols[0].set(qn('w:space'), '720') # 0.5 inch gap
    else:
        new_cols = OxmlElement('w:cols')
        new_cols.set(qn('w:num'), '2')
        new_cols.set(qn('w:space'), '720')
        sectPr.append(new_cols)

    def add_sec_heading(title):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(title)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(9.5)
        r.font.bold = True

    def add_subsec_heading(title):
        p = doc.add_paragraph()
        r = p.add_run(title)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.italic = True

    def add_body(text, indent=True):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if indent:
            p.paragraph_format.first_line_indent = Inches(0.18)
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(8.5)

    def add_caption(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(7.5)
        r.font.italic = True

    # Body sections
    add_sec_heading("I. INTRODUCTION")
    add_body("Industrial manufacturing relies heavily on computerized numerical control (CNC) milling machinery, automated tooling spindles, and multi-axis machining centers. Equipment breakdowns incur severe financial and operational losses, costing industrial facilities an estimated $50 billion annually in unplanned downtime [3], [4]. Historically, manufacturing facilities adopted reactive 'run-to-failure' strategies or rigid time-based preventative schedules [3]. While reactive repair creates unacceptable emergency outages, preventative maintenance frequently causes premature component replacement, discarding functional cutting heads with substantial useful life remaining [4].")
    add_body("Predictive maintenance (PdM) leverages multivariate sensor telemetry—such as spindle torque, cutting contact duration, rotational velocities, and thermal dissipation gradients—to dynamically forecast impending failure states [1], [5]. However, real-world industrial adoption faces the critical 'black-box dilemma' [6], [7]. Advanced deep learning and tree ensemble algorithms output probabilistic risk scores without exposing the underlying physical drivers. If an autonomous model flags an alert without physical attribution, plant operators cannot ascertain whether the alarm stems from thermal exhaustion, mechanical overstrain, or electrical fluctuation [7]. Under high-stakes operational pressure, uninterpretable alerts lead to either costly unnecessary line halts or dangerous alarm dismissal.")
    add_body("Furthermore, predictive maintenance is governed by profound misclassification cost asymmetry. Missing an actual machine failure (False Negative) leads to catastrophic spindle destruction, ruined workpieces, and collateral assembly line stoppage, with remediation costs often exceeding $10,000 to $100,000 [3]. Conversely, a false warning (False Positive) incurs only a brief 10-minute diagnostic inspection ($100–$500). Consequently, standard classification accuracy is an inappropriate metric under severe minority failure distributions, demanding optimization focused heavily on Failure Recall [6].")
    add_body("To address these challenges, we present X-Maintain, an Explainable AI predictive maintenance framework pairing high-recall gradient boosting with exact game-theoretic Shapley attributions (TreeSHAP) [2], [12].")

    add_sec_heading("II. LITERATURE REVIEW")
    add_subsec_heading("A. Predictive Maintenance in Industry 4.0")
    add_body("Predictive maintenance has transitioned from offline vibration spectral analysis toward real-time IoT telemetry analytics [3], [4]. Zonta et al. [4] reviewed over 100 industrial implementations, finding condition monitoring reduces unexpected outages by up to 45% and lowers maintenance costs by 25%. However, Dalzochio et al. [5] highlighted that industrial telemetry is frequently beset by severe class imbalance, where healthy states represent over 95% of recorded instances. In milling, failure mechanisms are governed by physical interactions: tool wear obeys Taylor's tool life model, power consumption relates to torque and angular velocity, and thermal dissipation governs heat transfer between workpiece and tool [1], [3].")
    add_subsec_heading("B. Machine Learning Algorithms for PdM")
    add_body("Ensemble learning consistently demonstrates superiority over standard linear and kernel classifiers on industrial tabular datasets [3]. Chen and Guestrin [8] established Extreme Gradient Boosting (XGBoost), employing second-order Taylor loss expansions and column subsampling. Breiman [9] formulated Random Forests, leveraging bootstrap aggregation to mitigate variance. Carvalho et al. [3] documented that tree ensembles achieve superior F1 scores on mechanical telemetry compared to support vector machines and neural networks, primarily because decision trees naturally capture step-function thresholds inherent to physical safety boundaries.")
    add_subsec_heading("C. Class Imbalance Mitigation Strategies")
    add_body("Under severe class imbalance, unweighted empirical risk minimization biases classifiers toward the majority class [10]. Chawla et al. [10] developed SMOTE, synthesizing minority points along k-nearest neighbors. However, synthetic interpolation in physical thermodynamic spaces risks producing unphysical sensor combinations. In contrast, cost-sensitive learning adjusts loss penalties via positive class weighting, penalizing minority false negatives without distorting feature geometry [8]. Post-hoc threshold tuning optimizes the classification boundary along the validation Precision-Recall curve.")
    add_subsec_heading("D. Explainable AI and Shapley Attributions")
    add_body("Ribeiro et al. [11] proposed LIME, using local surrogate linear approximations. However, LIME lacks mathematical consistency and exhibits high sampling variance across identical inputs. Lundberg and Lee [2] formulated SHAP, grounded in cooperative game theory [2]. Lundberg et al. [12] introduced TreeSHAP, enabling polynomial-time exact computation of Shapley values for decision trees. Steurtewagen and Van den Poel [6] and Gawde et al. [7] demonstrated that SHAP reveals critical degradation thresholds in rotating machinery, bridging the trust gap for operations personnel.")

    add_sec_heading("III. DATASET AND PROBLEM FORMULATION")
    add_subsec_heading("A. Dataset Overview and Synthetic Origin")
    add_body("We utilize the AI4I 2020 Predictive Maintenance Dataset [1], hosted by the UCI Machine Learning Repository. The dataset contains 10,000 operational records reflecting synthetic milling machine telemetry generated from physical simulation models [1]. AI4I 2020 is a synthetic benchmark designed to mirror industrial cutting dynamics without sensor dropouts. Each record describes an individual machining process cycle for a tool workpiece.")
    add_subsec_heading("B. Physical Failure Modes")
    add_body("The simulation encodes five distinct physical failure mechanisms [1]: 1) Tool Wear Failure (TWF): cutter wear exceeding 200–240 minutes; 2) Heat Dissipation Failure (HDF): temperature difference Delta T < 8.6 K with spindle speed < 1,380 rpm; 3) Power Failure (PWF): mechanical power P = tau * omega outside 3,500 W to 9,000 W; 4) Overstrain Failure (OSF): torque times tool wear exceeding variant thresholds (11,000 to 13,000 min*Nm); 5) Random Failure (RNF): 0.1% background defects.")
    add_subsec_heading("C. Target Leakage Prevention")
    add_body("The raw dataset records the five failure-mode indicators (TWF, HDF, PWF, OSF, RNF). Crucially, the target variable is a logical union of these modes: Machine failure = TWF or HDF or PWF or OSF or RNF. Retaining any failure-mode column provides the model with direct target leakage, trivially inflating accuracy to 99.9% while rendering the model useless for proactive forecasting. X-Maintain strictly purges all five failure mode columns as well as arbitrary primary identifiers (UDI, Product ID).")

    add_sec_heading("IV. PROPOSED X-MAINTAIN FRAMEWORK")
    add_body("The X-Maintain framework comprises five modular layers: 1) Ingestion and Leakage Purge, 2) Stratified Preprocessing Pipeline, 3) Cost-Sensitive Multi-Model Training, 4) Game-Theoretic TreeSHAP Attribution, and 5) Interactive Streamlit Deployment with What-If Sensitivity Simulation. Preprocessing enforces stratified 70/10/20 partitioning (6,999 train, 1,000 val, 2,001 test). StandardScaler is fitted exclusively on training numerical features. The categorical Type variable is one-hot encoded into three binary indicators.")

    add_sec_heading("V. MACHINE LEARNING MODELS")
    add_body("Three models are formulated: 1) Balanced Logistic Regression establishing linear baseline; 2) Balanced Random Forest (200 trees); 3) Cost-Sensitive XGBoost configured with scale_pos_weight = 28.53 (6,762 negative / 237 positive training instances). Model selection is governed by a domain utility metric: Utility = 0.4*Recall + 0.3*F1 + 0.3*ROC-AUC.")

    add_sec_heading("VI. EXPERIMENTAL RESULTS AND BENCHMARK")
    add_body("On the held-out test split (N = 2,001, 68 true breakdowns), XGBoost achieves the top selection score (0.8064), balancing 73.53% Recall (50/68 failures captured) with 73.53% Precision (only 18 false alarms), 98.20% accuracy, 0.9724 ROC-AUC, and 0.7934 PR-AUC. In comparison, Logistic Regression achieves 79.41% recall but precision collapses to 14.67% (314 false alarms). Random Forest achieves 87.10% precision but misses over 60% of true failure events (Recall = 39.71%, 41 missed failures).")

    add_sec_heading("VII. SHAP EXPLAINABILITY AND SENSITIVITY ENGINE")
    add_body("TreeSHAP analysis confirms Torque (mean |SHAP| = 3.74) and Tool Wear (3.06) dominate failure predictions, followed by Rotational Speed (2.07) and Air Temperature (2.01). Beeswarm and dependence plots show non-linear inflection: torque above 50 Nm and tool wear above 200 min sharply escalate failure log-odds. Local waterfall attributions for breakdown instance #11 decompose log-odds from -3.76 base expectation to +11.49 prediction, with Torque (+7.37) and Tool Wear (+4.19) as dominant drivers.")

    add_sec_heading("VIII. SOFTWARE VALIDATION AND TESTING")
    add_body("The system is verified through an automated 20-test Pytest suite covering preprocessing integrity, prediction shape and probability bounds [0, 1], and TreeExplainer tensor operations, passing 20/20 tests in 3.21 seconds.")

    add_sec_heading("IX. CONCLUSION")
    add_body("X-Maintain demonstrates that combining leakage-free preprocessing, cost-sensitive gradient boosting, and exact TreeSHAP attribution produces a dependable, interpretable predictive maintenance framework suitable for high-stakes industrial deployments.")

    add_sec_heading("REFERENCES")
    refs = [
        "[1] S. Matzka, \"Explainable Artificial Intelligence for Predictive Maintenance Applications,\" in Proc. IEEE AI4I, 2020, pp. 69–74.",
        "[2] S. M. Lundberg and S.-I. Lee, \"A Unified Approach to Interpreting Model Predictions,\" in Proc. NeurIPS, 2017, pp. 4765–4774.",
        "[3] T. P. Carvalho et al., \"A systematic literature review of machine learning methods applied to predictive maintenance,\" Comput. Ind. Eng., vol. 137, p. 106024, 2019.",
        "[4] T. Zonta et al., \"Predictive maintenance in the Industry 4.0: A systematic literature review,\" Comput. Ind. Eng., vol. 150, p. 106889, 2020.",
        "[5] J. Dalzochio et al., \"Machine learning and reasoning for predictive maintenance in Industry 4.0: Current status and challenges,\" Comput. Ind., vol. 123, p. 103298, 2020.",
        "[6] E. Steurtewagen and D. Van den Poel, \"Adding interpretability to predictive maintenance by machine learning on sensor data,\" Comput. Ind. Eng., vol. 152, p. 107047, 2021.",
        "[7] P. Gawde et al., \"Explainable Predictive Maintenance of Rotating Machines Using LIME, SHAP, PDP, ICE,\" in Proc. IEEE BigData, 2021, pp. 3122–3131.",
        "[8] T. Chen and C. Guestrin, \"XGBoost: A Scalable Tree Boosting System,\" in Proc. ACM KDD, 2016, pp. 785–794.",
        "[9] L. Breiman, \"Random Forests,\" Mach. Learn., vol. 45, no. 1, pp. 5–32, 2001.",
        "[10] N. V. Chawla et al., \"SMOTE: Synthetic Minority Over-sampling Technique,\" J. Artif. Intell. Res., vol. 16, pp. 321–357, 2002.",
        "[11] M. T. Ribeiro et al., \"Why Should I Trust You?: Explaining the Predictions of Any Classifier,\" in Proc. ACM KDD, 2016, pp. 1135–1144.",
        "[12] S. M. Lundberg et al., \"From local explanations to global understanding with explainable AI for trees,\" Nat. Mach. Intell., vol. 2, pp. 56–67, 2020."
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r_run = p.add_run(r)
        r_run.font.name = 'Times New Roman'
        r_run.font.size = Pt(7.5)

    doc.save(docx_path)
    print(f"Word document generated successfully at {docx_path}!")

# =========================================================================
# 3. LATEX SOURCE GENERATION
# =========================================================================

def build_latex():
    print("Building LaTeX source document...")
    tex_path = os.path.join(BASE_DIR, "X_Maintain_IEEE_Research_Paper.tex")

    tex_content = r"""\documentclass[journal,10pt,twocolumn]{IEEEtran}

\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{cite}
\usepackage{url}
\usepackage{microtype}

\begin{document}

\title{X-Maintain: An Explainable Machine Learning Framework for Predictive Maintenance and Machine Failure Prediction}

\author{Syagamreddy~Gopi~Adithya~Vardhan~Reddy%
\thanks{S. G. A. V. Reddy is with the Department of Computer Science and Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu 632014, India (e-mail: gopi.adithya2022@vitstudent.ac.in).}}

\markboth{IEEE Transactions on Industrial Informatics (Special Section on Explainable AI)}%
{Reddy: X-Maintain Explainable Machine Learning Framework for Predictive Maintenance}

\maketitle

\begin{abstract}
Industrial predictive maintenance (PdM) relies on multivariate sensor streams to anticipate equipment failure before catastrophic downtime occurs. However, contemporary high-performance machine learning ensembles operate as opaque black boxes, creating an operational trust barrier for factory engineers who require actionable root-cause insights before halting costly production spindles. This paper presents X-Maintain, an end-to-end explainable artificial intelligence (XAI) framework for predictive maintenance. Using the benchmark AI4I 2020 Predictive Maintenance Dataset (10,000 milling operations), we enforce strict target leakage prevention by excluding failure-mode indicators (TWF, HDF, PWF, OSF, RNF) and arbitrary serial keys, yielding an eight-feature operational input matrix. We benchmark three classifier families---Logistic Regression, Random Forest, and Extreme Gradient Boosting (XGBoost)---and systematically evaluate four class-imbalance mitigation strategies under severe 28.5:1 class disparity. Model selection is governed by a domain-grounded utility metric (0.4$\times$Recall + 0.3$\times$F1 + 0.3$\times$ROC-AUC). On a held-out test split ($N = 2,001$), cost-sensitive XGBoost achieves 98.20\% accuracy, 73.53\% precision, 73.53\% failure recall, 0.9724 ROC-AUC, and 0.7934 PR-AUC. Model decisions are interpreted using TreeSHAP, revealing that Torque (mean $|$SHAP$| = 3.74$) and Tool Wear (mean $|$SHAP$| = 3.06$) dominate failure predictions. Local waterfall attributions, a deterministic hallucination-free natural language explanation generator, a model-based what-if sensitivity engine, and an enterprise Streamlit monitoring dashboard complete the validated architecture, backed by 20 passed unit tests.
\end{abstract}

\begin{IEEEkeywords}
Predictive maintenance, Explainable Artificial Intelligence (XAI), SHapley Additive exPlanations (SHAP), machine failure prediction, XGBoost, class imbalance, TreeExplainer, industrial IoT, model-based sensitivity analysis.
\end{IEEEkeywords}

\section{Introduction}
\IEEEPARstart{I}{ndustrial} manufacturing relies heavily on computerized numerical control (CNC) milling machinery, automated tooling spindles, and multi-axis machining centers. Equipment breakdowns incur severe financial and operational losses, costing industrial facilities an estimated \$50 billion annually in unplanned downtime \cite{carvalho2019, zonta2020}. Historically, manufacturing facilities adopted reactive ``run-to-failure'' strategies or rigid time-based preventative schedules \cite{carvalho2019}. While reactive repair creates unacceptable emergency outages, preventative maintenance frequently causes premature component replacement, discarding functional cutting heads with substantial useful life remaining \cite{zonta2020}.

Predictive maintenance (PdM) leverages multivariate sensor telemetry---such as spindle torque, cutting contact duration, rotational velocities, and thermal dissipation gradients---to dynamically forecast impending failure states \cite{matzka2020, dalzochio2020}. However, real-world industrial adoption faces the critical ``black-box dilemma'' \cite{steurtewagen2021, gawde2021}. Advanced deep learning and tree ensemble algorithms output probabilistic risk scores without exposing the underlying physical drivers. If an autonomous model flags an alert without physical attribution, plant operators cannot ascertain whether the alarm stems from thermal exhaustion, mechanical overstrain, or electrical fluctuation \cite{gawde2021}. Under high-stakes operational pressure, uninterpretable alerts lead to either costly unnecessary line halts or dangerous alarm dismissal.

Furthermore, predictive maintenance is governed by profound misclassification cost asymmetry. Missing an actual machine failure (False Negative) leads to catastrophic spindle destruction, ruined workpieces, and collateral assembly line stoppage, with remediation costs often exceeding \$10,000 to \$100,000 \cite{carvalho2019}. Conversely, a false warning (False Positive) incurs only a brief 10-minute diagnostic inspection (\$100--\$500). Consequently, standard classification accuracy is an inappropriate metric under severe minority failure distributions, demanding optimization focused heavily on Failure Recall \cite{steurtewagen2021}.

To address these challenges, we present \emph{X-Maintain}, an Explainable AI predictive maintenance framework that pairs high-recall gradient boosting with exact game-theoretic Shapley attributions (TreeSHAP) \cite{lundberg2017, lundberg2020}. The primary verified contributions of this work are:
\begin{enumerate}
    \item \emph{Leakage-Aware Preprocessing:} A rigorous data pipeline that purges downstream failure modes (TWF, HDF, PWF, OSF, RNF) and primary serial keys to guarantee academic and operational integrity.
    \item \emph{Empirical Imbalance Benchmarking:} A comparative evaluation across baseline Logistic Regression, Random Forest, and XGBoost under cost-sensitive weighting, SMOTE, and validation-tuned decision thresholds.
    \item \emph{Multi-Objective Model Selection:} A weighted utility formulation ($0.4\times\text{Recall} + 0.3\times\text{F1} + 0.3\times\text{ROC-AUC}$) optimizing minority failure capture while suppressing alarm fatigue.
    \item \emph{Dual-Scale TreeSHAP Explainability:} Exact global attribution across the dataset manifold and local waterfall decomposition for individual machine predictions.
    \item \emph{Model-Based What-If Analysis:} A counterfactual parameter perturbation engine allowing operators to simulate risk reduction under simulated operational adjustments without claiming physical causality.
    \item \emph{Production Deployment \& Verification:} An enterprise-grade 7-view Streamlit industrial monitoring interface validated through an automated 20-test Pytest suite.
\end{enumerate}

\section{Literature Review}
\subsection{Predictive Maintenance in Industry 4.0}
Predictive maintenance has transitioned from offline vibration spectral analysis toward real-time IoT telemetry analytics \cite{carvalho2019, zonta2020}. Zonta et al. \cite{zonta2020} reviewed over 100 industrial implementations, finding that sensor-driven condition monitoring reduces unexpected outages by up to 45\% and lowers maintenance costs by 25\%. However, Dalzochio et al. \cite{dalzochio2020} highlighted that industrial telemetry is frequently beset by class imbalance, where healthy states represent over 95\% of recorded instances. In CNC milling, failure mechanisms are governed by physical cutting interactions: tool flank wear obeys Taylor's tool life equation, power consumption relates directly to torque and angular velocity, and thermal dissipation governs heat transfer between workpiece and tool \cite{matzka2020, carvalho2019}.

\subsection{Machine Learning Algorithms for PdM}
Ensemble learning consistently demonstrates superiority over standard linear and kernel classifiers on industrial tabular datasets \cite{carvalho2019}. Chen and Guestrin \cite{chen2016} established Extreme Gradient Boosting (XGBoost), which employs second-order Taylor loss expansions and column subsampling. Breiman \cite{breiman2001} formulated Random Forests, leveraging bootstrap aggregation to mitigate variance. Carvalho et al. \cite{carvalho2019} documented that tree ensembles achieve superior F1 scores on mechanical telemetry compared to support vector machines and feedforward neural networks, primarily because decision trees naturally capture step-function thresholds inherent to physical safety boundaries.

\subsection{Class Imbalance Mitigation Strategies}
Under severe class imbalance, unweighted empirical risk minimization biases classifiers toward the majority class \cite{chawla2002}. Chawla et al. \cite{chawla2002} developed SMOTE, synthesizing minority points along k-nearest neighbors. However, synthetic interpolation in physical thermodynamic spaces risks producing unphysical sensor combinations. In contrast, cost-sensitive learning adjusts loss penalties via positive class weighting, penalizing minority false negatives without distorting feature geometry \cite{chen2016}. Post-hoc threshold tuning optimizes the classification boundary along the validation Precision-Recall curve.

\subsection{Explainable AI and Shapley Attributions}
Ribeiro et al. \cite{ribeiro2016} proposed LIME, using local surrogate linear approximations. However, LIME lacks mathematical consistency and exhibits high sampling variance across identical inputs. Lundberg and Lee \cite{lundberg2017} formulated SHAP, grounded in cooperative game theory. Lundberg et al. \cite{lundberg2020} introduced TreeSHAP, enabling polynomial-time $O(TLD^2)$ exact computation of Shapley values for decision trees. Steurtewagen and Van den Poel \cite{steurtewagen2021} and Gawde et al. \cite{gawde2021} demonstrated that SHAP reveals critical degradation thresholds in rotating machinery, bridging the trust gap for operations personnel.

\begin{table*}[t]
\centering
\caption{Comparative Analysis of Related Work in Predictive Maintenance and Explainable AI}
\label{tab:related_work}
\begin{tabular}{llllll}
\toprule
\textbf{Reference} & \textbf{Year} & \textbf{Algorithm} & \textbf{XAI Method} & \textbf{Core Contribution} & \textbf{Methodological Limitation / Research Gap} \\
\midrule
Matzka \cite{matzka2020} & 2020 & RF / kNN & SHAP / PDP & Introduction of AI4I benchmark & Retained downstream failure mode leakage columns \\
Lundberg \cite{lundberg2017} & 2017 & General & SHAP & Unified game-theoretic XAI formulation & Pure theoretical framework; no industrial pipeline \\
Carvalho \cite{carvalho2019} & 2019 & Survey & None & Systematic review of ML in PdM & Lacks evaluation of model interpretability \\
Zonta \cite{zonta2020} & 2020 & Survey & None & Industry 4.0 PdM application survey & No post-hoc XAI architectural integration \\
Dalzochio \cite{dalzochio2020} & 2020 & Survey & Rule-based & ML and reasoning for PdM & Limited to expert systems; lacks modern tree XAI \\
Steurtewagen \cite{steurtewagen2021} & 2021 & RF / Logit & SHAP & Sensor interpretability in manufacturing & No model-based what-if sensitivity engine \\
Gawde \cite{gawde2021} & 2021 & XGBoost & SHAP / LIME & Rotating machinery XAI study & Lacks production deployment interface \\
Chen \cite{chen2016} & 2016 & XGBoost & Split Gain & Scalable tree boosting architecture & Black-box predictions without additive attributions \\
Breiman \cite{breiman2001} & 2001 & RF & Impurity & Bagged ensemble decision trees & Gini importance exhibits strong cardinality bias \\
Chawla \cite{chawla2002} & 2002 & SMOTE & None & Synthetic minority over-sampling & Synthetic interpolation distorts physical geometries \\
Lundberg \cite{lundberg2020} & 2020 & TreeSHAP & TreeSHAP & Polynomial-time exact tree explanations & Algorithm presentation without PdM workflow \\
\textbf{This Work} & \textbf{2026} & \textbf{XGBoost} & \textbf{TreeSHAP} & \textbf{End-to-end explainable PdM framework} & \textbf{Fully integrated leakage-free production pipeline} \\
\bottomrule
\end{tabular}
\end{table*}

\section{Dataset and Problem Formulation}
\subsection{Dataset Overview and Synthetic Origin}
We utilize the AI4I 2020 Predictive Maintenance Dataset \cite{matzka2020}, hosted by the UCI Machine Learning Repository. The dataset contains 10,000 operational records reflecting synthetic milling machine telemetry generated from physical simulation models \cite{matzka2020}. It is essential to clarify that AI4I 2020 is a synthetic benchmark designed to mirror industrial cutting dynamics without sensor dropouts, rather than raw field-harvested factory telemetry.

\subsection{Physical Failure Modes}
The simulation encodes five distinct physical failure mechanisms \cite{matzka2020}:
\begin{enumerate}
    \item \emph{Tool Wear Failure (TWF):} Cutter contact friction gradually wears down the milling tool. Based on Taylor's tool life model, failure occurs when cumulative contact duration satisfies $\text{Tool wear} \ge [200, 240]$ minutes.
    \item \emph{Heat Dissipation Failure (HDF):} Heat generated during cutting must be dissipated into the surrounding atmosphere. When the temperature difference $\Delta T = T_{\text{process}} - T_{\text{air}} < 8.6\text{ K}$ while rotational speed $\omega < 1,380\text{ rpm}$, cooling airflow is insufficient, triggering thermal seizure.
    \item \emph{Power Failure (PWF):} The mechanical power required for machining is defined by $P = \tau \times \frac{2\pi\omega}{60}$, where $\tau$ is torque in Nm and $\omega$ is rotational speed in rpm. Power failure occurs when $P < 3,500\text{ W}$ (stalling) or $P > 9,000\text{ W}$ (overload).
    \item \emph{Overstrain Failure (OSF):} The product of tool wear and torque reflects mechanical stress intensity. When $\tau \times [\text{Tool wear}]$ exceeds variant thresholds (11,000 $\text{min}\cdot\text{Nm}$ for Type L, 12,000 for Type M, and 13,000 for Type H), the tool fractures.
    \item \emph{Random Failure (RNF):} Unforeseen mechanical defects occur stochastically with a 0.1\% background failure probability.
\end{enumerate}

\subsection{Target Leakage Prevention}
The raw dataset records the five failure-mode indicators (TWF, HDF, PWF, OSF, RNF). Crucially, the target variable is a logical union of these modes: $\text{Machine failure} = \text{TWF} \lor \text{HDF} \lor \text{PWF} \lor \text{OSF} \lor \text{RNF}$. Retaining any failure-mode column provides the model with direct target leakage, trivially inflating accuracy to 99.9\% while rendering the model useless for proactive forecasting. X-Maintain strictly purges all five failure mode columns as well as arbitrary primary identifiers (UDI, Product ID).

\section{Proposed X-Maintain Framework}
The architecture comprises five modular layers: 1) Ingestion and Leakage Purge, 2) Stratified Preprocessing Pipeline, 3) Cost-Sensitive Multi-Model Training, 4) Game-Theoretic TreeSHAP Attribution, and 5) Interactive Streamlit Deployment with What-If Sensitivity Simulation.

Data partitioning enforces stratified sampling with a fixed seed ($\text{RANDOM\_STATE} = 42$): 70\% Train (6,999 records, 237 failures), 10\% Validation (1,000 records, 34 failures), and 20\% Test (2,001 records, 68 failures). StandardScaler is fitted exclusively on training numerical features and applied to validation/test partitions. The categorical Type variable is one-hot encoded into Type\_H, Type\_L, and Type\_M.

\section{Machine Learning Models}
\subsection{Cost-Sensitive XGBoost}
XGBoost constructs additive regression trees minimizing regularized objective via second-order Taylor expansion \cite{chen2016}:
\begin{equation}
\tilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2
\end{equation}
Dynamic cost-sensitive weighting is configured via:
\begin{equation}
\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{6,762}{237} = 28.53
\end{equation}
penalizing false negatives 28.53 times more heavily than false positives.

\subsection{TreeSHAP Attribution Formulation}
Shapley values allocate additive feature contributions relative to base expected output \cite{lundberg2017, lundberg2020}:
\begin{equation}
\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]
\end{equation}
satisfying efficiency, symmetry, dummy, and additivity axioms \cite{lundberg2020}.

\section{Experimental Results}
Table \ref{tab:results} summarizes the held-out test evaluation ($N = 2,001$, 68 true breakdowns). XGBoost achieves top utility (0.8064), balancing 73.53\% Recall with 73.53\% Precision and 0.9724 ROC-AUC.

\begin{table}[b]
\centering
\caption{Model Performance Benchmark on Held-Out Test Split ($N = 2,001$)}
\label{tab:results}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Acc.} & \textbf{Prec.} & \textbf{Recall} & \textbf{F1} & \textbf{ROC} & \textbf{Score} \\
\midrule
Logistic Reg. & 83.61\% & 14.67\% & \textbf{79.41\%} & 24.77\% & 0.8949 & 0.6604 \\
Random Forest & 97.75\% & \textbf{87.10\%} & 39.71\% & 54.55\% & 0.9556 & 0.6091 \\
\textbf{XGBoost} & \textbf{98.20\%} & 73.53\% & 73.53\% & \textbf{73.53\%} & \textbf{0.9724} & \textbf{0.8064} \\
\bottomrule
\end{tabular}
\end{table}

\section{Explainability Analysis}
TreeSHAP reveals that Torque ($\text{mean } |\text{SHAP}| = 3.74$) and Tool Wear (3.06) dominate failure predictions, followed by Rotational Speed (2.07) and Air Temperature (2.01). Dependence plots show sharp inflection above 50 Nm and 200 min cutter degradation.

\section{Software Validation}
The framework is verified by an automated 20-test Pytest suite across preprocessing, prediction bounds, and TreeExplainer tensor operations, achieving a 100\% pass rate in 3.21 seconds.

\section{Conclusion}
X-Maintain provides an auditable, transparent operational bridge between predictive machine learning and industrial maintenance decision support.

\begin{thebibliography}{00}
\bibitem{matzka2020} S. Matzka, ``Explainable Artificial Intelligence for Predictive Maintenance Applications,'' in \emph{Proc. IEEE AI4I}, 2020, pp. 69--74.
\bibitem{lundberg2017} S. M. Lundberg and S.-I. Lee, ``A Unified Approach to Interpreting Model Predictions,'' in \emph{Proc. NeurIPS}, 2017, pp. 4765--4774.
\bibitem{carvalho2019} T. P. Carvalho et al., ``A systematic literature review of machine learning methods applied to predictive maintenance,'' \emph{Comput. Ind. Eng.}, vol. 137, p. 106024, 2019.
\bibitem{zonta2020} T. Zonta et al., ``Predictive maintenance in the Industry 4.0: A systematic literature review,'' \emph{Comput. Ind. Eng.}, vol. 150, p. 106889, 2020.
\bibitem{dalzochio2020} J. Dalzochio et al., ``Machine learning and reasoning for predictive maintenance in Industry 4.0: Current status and challenges,'' \emph{Comput. Ind.}, vol. 123, p. 103298, 2020.
\bibitem{steurtewagen2021} E. Steurtewagen and D. Van den Poel, ``Adding interpretability to predictive maintenance by machine learning on sensor data,'' \emph{Comput. Ind. Eng.}, vol. 152, p. 107047, 2021.
\bibitem{gawde2021} P. Gawde et al., ``Explainable Predictive Maintenance of Rotating Machines Using LIME, SHAP, PDP, ICE,'' in \emph{Proc. IEEE BigData}, 2021, pp. 3122--3131.
\bibitem{chen2016} T. Chen and C. Guestrin, ``XGBoost: A Scalable Tree Boosting System,'' in \emph{Proc. ACM KDD}, 2016, pp. 785--794.
\bibitem{breiman2001} L. Breiman, ``Random Forests,'' \emph{Mach. Learn.}, vol. 45, no. 1, pp. 5--32, 2001.
\bibitem{chawla2002} N. V. Chawla et al., ``SMOTE: Synthetic Minority Over-sampling Technique,'' \emph{J. Artif. Intell. Res.}, vol. 16, pp. 321--357, 2002.
\bibitem{ribeiro2016} M. T. Ribeiro et al., ``"Why Should I Trust You?": Explaining the Predictions of Any Classifier,'' in \emph{Proc. ACM KDD}, 2016, pp. 1135--1144.
\bibitem{lundberg2020} S. M. Lundberg et al., ``From local explanations to global understanding with explainable AI for trees,'' \emph{Nat. Mach. Intell.}, vol. 2, pp. 56--67, 2020.
\end{thebibliography}

\end{document}
"""
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_content.strip())
    print(f"LaTeX document generated successfully at {tex_path}!")

if __name__ == '__main__':
    page_count = build_pdf()
    build_docx()
    build_latex()
    print(f"\n==========================================")
    print(f"ALL DELIVERABLES GENERATED SUCCESSFULLY!")
    print(f"1. PDF:   X_Maintain_IEEE_Research_Paper.pdf ({page_count} pages)")
    print(f"2. DOCX:  X_Maintain_IEEE_Research_Paper.docx")
    print(f"3. LATEX: X_Maintain_IEEE_Research_Paper.tex")
    print(f"==========================================")

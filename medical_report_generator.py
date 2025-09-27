# medical_report_generator.py - Professional Medical Report Generator for V-CARE

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph, 
                               Spacer, PageBreak, Image as RLImage)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib.colors import HexColor
import os
import tempfile
from io import BytesIO
import base64
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ClinicalReportGenerator:
    """Professional medical report generator for aerospace cognitive wellness data."""
    
    def __init__(self, reports_dir: str = "clinical_reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        
        # Medical color scheme
        self.colors = {
            'primary': HexColor('#1a365d'),
            'secondary': HexColor('#2c5282'), 
            'accent': HexColor('#3182ce'),
            'success': HexColor('#38a169'),
            'warning': HexColor('#d69e2e'),
            'danger': HexColor('#e53e3e'),
            'neutral': HexColor('#4a5568'),
            'light_gray': HexColor('#f7fafc'),
            'med_gray': HexColor('#edf2f7')
        }
        
        # Medical reference ranges (based on aerospace medicine standards)
        self.medical_ranges = {
            'cwi': {'optimal': (85, 100), 'good': (70, 84), 'fair': (55, 69), 'poor': (40, 54), 'critical': (0, 39)},
            'stress': {'normal': (0, 30), 'mild': (31, 50), 'moderate': (51, 70), 'high': (71, 85), 'severe': (86, 100)},
            'fatigue': {'normal': (0, 25), 'mild': (26, 40), 'moderate': (41, 60), 'high': (61, 80), 'severe': (81, 100)},
            'blink_rate': {'normal': (12, 25), 'mild_fatigue': (26, 35), 'moderate_fatigue': (36, 45), 'high_fatigue': (46, 60)}
        }
    
    def generate_clinical_report(self, astronaut_data: Dict, mission_data: Dict, 
                               wellness_timeline: List[Dict], alerts: List[Dict],
                               report_period: Tuple[datetime, datetime]) -> str:
        """Generate comprehensive clinical report."""
        
        report_filename = f"Clinical_Report_{mission_data['mission_code']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = os.path.join(self.reports_dir, report_filename)
        
        # Create the document
        doc = SimpleDocTemplate(report_path, pagesize=A4, rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Build story
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=self.colors['primary']
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=12,
            textColor=self.colors['secondary']
        )
        
        # Title page
        story.extend(self._create_title_page(astronaut_data, mission_data, report_period, title_style, styles))
        story.append(PageBreak())
        
        # Executive Summary
        story.append(Paragraph("EXECUTIVE SUMMARY", header_style))
        story.extend(self._create_executive_summary(wellness_timeline, alerts, styles))
        story.append(Spacer(1, 20))
        
        # Medical Assessment
        story.append(Paragraph("MEDICAL ASSESSMENT", header_style))
        story.extend(self._create_medical_assessment(wellness_timeline, astronaut_data, styles))
        story.append(Spacer(1, 20))
        
        # Physiological Analysis
        story.append(Paragraph("PHYSIOLOGICAL ANALYSIS", header_style))
        story.extend(self._create_physiological_analysis(wellness_timeline, styles))
        story.append(PageBreak())
        
        # Data Visualizations
        story.append(Paragraph("CLINICAL DATA VISUALIZATION", header_style))
        story.extend(self._create_data_visualizations(wellness_timeline, styles))
        story.append(PageBreak())
        
        # Alert Analysis
        story.append(Paragraph("ALERT AND INCIDENT ANALYSIS", header_style))
        story.extend(self._create_alert_analysis(alerts, styles))
        story.append(Spacer(1, 20))
        
        # Clinical Recommendations
        story.append(Paragraph("CLINICAL RECOMMENDATIONS", header_style))
        story.extend(self._create_clinical_recommendations(wellness_timeline, alerts, astronaut_data, styles))
        story.append(PageBreak())
        
        # Detailed Data Tables
        story.append(Paragraph("DETAILED PHYSIOLOGICAL DATA", header_style))
        story.extend(self._create_detailed_data_tables(wellness_timeline, styles))
        
        # Footer
        story.extend(self._create_footer(styles))
        
        # Build the PDF
        doc.build(story)
        
        logger.info(f"Clinical report generated: {report_path}")
        return report_path
    
    def _create_title_page(self, astronaut_data: Dict, mission_data: Dict, 
                          report_period: Tuple[datetime, datetime], title_style, styles) -> List:
        """Create professional title page."""
        content = []
        
        # Main title
        content.append(Paragraph("COGNITIVE WELLNESS CLINICAL REPORT", title_style))
        content.append(Spacer(1, 30))
        
        # Report classification
        classification_style = ParagraphStyle(
            'Classification',
            parent=styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            textColor=colors.red,
            fontName='Helvetica-Bold'
        )
        content.append(Paragraph("MEDICAL CONFIDENTIAL - FOR AUTHORIZED PERSONNEL ONLY", classification_style))
        content.append(Spacer(1, 40))
        
        # Patient information table
        patient_data = [
            ['ASTRONAUT INFORMATION', ''],
            ['Astronaut ID:', astronaut_data['id']],
            ['Full Name:', astronaut_data['full_name']],
            ['Mission Code:', mission_data['mission_code']],
            ['Mission Type:', mission_data['mission_type']],
            ['Mission Phase:', mission_data['current_phase']],
            ['Risk Level:', mission_data['risk_level'].upper()],
            ['', ''],
            ['REPORT PERIOD', ''],
            ['Start Date:', report_period[0].strftime('%Y-%m-%d %H:%M:%S UTC')],
            ['End Date:', report_period[1].strftime('%Y-%m-%d %H:%M:%S UTC')],
            ['Duration:', str(report_period[1] - report_period[0])],
            ['', ''],
            ['REPORT INFORMATION', ''],
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')],
            ['System:', 'V-CARE Cognitive Wellness Monitor v2.0'],
            ['Report Type:', 'Comprehensive Clinical Assessment']
        ]
        
        table = Table(patient_data, colWidths=[2.5*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 8), (-1, 8), self.colors['secondary']),
            ('TEXTCOLOR', (0, 8), (-1, 8), colors.whitesmoke),
            ('FONTNAME', (0, 8), (-1, 8), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 13), (-1, 13), self.colors['secondary']),
            ('TEXTCOLOR', (0, 13), (-1, 13), colors.whitesmoke),
            ('FONTNAME', (0, 13), (-1, 13), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        content.append(table)
        content.append(Spacer(1, 40))
        
        # Medical disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=9,
            alignment=TA_JUSTIFY,
            textColor=self.colors['neutral']
        )
        
        disclaimer_text = """
        <b>MEDICAL DISCLAIMER:</b> This report is generated by the V-CARE Cognitive Wellness Monitoring System 
        and is intended for use by qualified flight surgeons and aerospace medical personnel. The data presented 
        should be interpreted in conjunction with clinical judgment and other medical assessments. This system 
        provides continuous monitoring data for cognitive wellness trends and should not replace comprehensive 
        medical evaluation.
        """
        
        content.append(Paragraph(disclaimer_text, disclaimer_style))
        
        return content
    
    def _create_executive_summary(self, wellness_timeline: List[Dict], alerts: List[Dict], styles) -> List:
        """Create executive summary with key findings."""
        content = []
        
        if not wellness_timeline:
            content.append(Paragraph("No wellness data available for analysis.", styles['Normal']))
            return content
        
        # Calculate summary statistics
        df = pd.DataFrame(wellness_timeline)
        
        summary_stats = {
            'avg_cwi': df['cognitive_wellness_index'].mean(),
            'min_cwi': df['cognitive_wellness_index'].min(),
            'max_cwi': df['cognitive_wellness_index'].max(),
            'avg_stress': df['stress_level'].mean(),
            'max_stress': df['stress_level'].max(),
            'avg_fatigue': df['fatigue_level'].mean(),
            'max_fatigue': df['fatigue_level'].max(),
            'total_records': len(df),
            'monitoring_hours': (pd.to_datetime(df['timestamp'].iloc[-1]) - 
                               pd.to_datetime(df['timestamp'].iloc[0])).total_seconds() / 3600
        }
        
        # Clinical assessment
        cwi_status = self._get_medical_classification(summary_stats['avg_cwi'], 'cwi')
        stress_status = self._get_medical_classification(summary_stats['max_stress'], 'stress')
        fatigue_status = self._get_medical_classification(summary_stats['max_fatigue'], 'fatigue')
        
        summary_text = f"""
        <b>CLINICAL OVERVIEW:</b><br/>
        During the {summary_stats['monitoring_hours']:.1f}-hour monitoring period, the astronaut's cognitive wellness 
        was continuously assessed through {summary_stats['total_records']} data points. The average Cognitive Wellness 
        Index (CWI) was {summary_stats['avg_cwi']:.1f}%, classified as <b>{cwi_status.upper()}</b>.<br/><br/>
        
        <b>KEY FINDINGS:</b><br/>
        • Maximum stress level reached {summary_stats['max_stress']:.1f}% ({stress_status})<br/>
        • Maximum fatigue level reached {summary_stats['max_fatigue']:.1f}% ({fatigue_status})<br/>
        • {len(alerts)} alerts generated during monitoring period<br/>
        • CWI range: {summary_stats['min_cwi']:.1f}% - {summary_stats['max_cwi']:.1f}%<br/><br/>
        
        <b>CLINICAL SIGNIFICANCE:</b><br/>
        {self._generate_clinical_significance(summary_stats, alerts)}
        """
        
        content.append(Paragraph(summary_text, styles['Normal']))
        
        # Summary statistics table
        content.append(Spacer(1, 20))
        
        summary_table_data = [
            ['PARAMETER', 'VALUE', 'CLINICAL RANGE', 'STATUS'],
            ['Average CWI', f"{summary_stats['avg_cwi']:.1f}%", "≥85% Optimal", cwi_status],
            ['Peak Stress Level', f"{summary_stats['max_stress']:.1f}%", "≤30% Normal", stress_status],
            ['Peak Fatigue Level', f"{summary_stats['max_fatigue']:.1f}%", "≤25% Normal", fatigue_status],
            ['Total Alerts', f"{len(alerts)}", "0 Ideal", "Review Required" if alerts else "Normal"],
            ['Monitoring Duration', f"{summary_stats['monitoring_hours']:.1f} hours", "Continuous", "Complete"]
        ]
        
        summary_table = Table(summary_table_data, colWidths=[2*inch, 1*inch, 1.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.colors['light_gray']])
        ]))
        
        content.append(summary_table)
        
        return content
    
    def _create_medical_assessment(self, wellness_timeline: List[Dict], astronaut_data: Dict, styles) -> List:
        """Create detailed medical assessment."""
        content = []
        
        if not wellness_timeline:
            content.append(Paragraph("Insufficient data for medical assessment.", styles['Normal']))
            return content
        
        df = pd.DataFrame(wellness_timeline)
        
        # Baseline comparison
        baseline_cwi = astronaut_data.get('baseline_cwi_score', 85.0)
        baseline_stress = astronaut_data.get('baseline_stress_threshold', 70.0)
        baseline_fatigue = astronaut_data.get('baseline_fatigue_threshold', 65.0)
        
        current_cwi = df['cognitive_wellness_index'].mean()
        current_stress = df['stress_level'].mean()
        current_fatigue = df['fatigue_level'].mean()
        
        assessment_text = f"""
        <b>BASELINE COMPARISON:</b><br/>
        The astronaut's current cognitive wellness metrics show the following deviations from established baselines:<br/><br/>
        
        • CWI: {current_cwi:.1f}% vs. baseline {baseline_cwi:.1f}% 
        ({'+' if current_cwi >= baseline_cwi else ''}{current_cwi - baseline_cwi:.1f}% change)<br/>
        • Stress: {current_stress:.1f}% vs. threshold {baseline_stress:.1f}% 
        ({'+' if current_stress >= baseline_stress else ''}{current_stress - baseline_stress:.1f}% change)<br/>
        • Fatigue: {current_fatigue:.1f}% vs. threshold {baseline_fatigue:.1f}% 
        ({'+' if current_fatigue >= baseline_fatigue else ''}{current_fatigue - baseline_fatigue:.1f}% change)<br/><br/>
        
        <b>TREND ANALYSIS:</b><br/>
        {self._analyze_trends(df)}
        
        <b>EMOTIONAL STABILITY:</b><br/>
        {self._analyze_emotional_patterns(df)}
        """
        
        content.append(Paragraph(assessment_text, styles['Normal']))
        
        return content
    
    def _create_physiological_analysis(self, wellness_timeline: List[Dict], styles) -> List:
        """Create physiological analysis section."""
        content = []
        
        if not wellness_timeline:
            content.append(Paragraph("No physiological data available.", styles['Normal']))
            return content
        
        df = pd.DataFrame(wellness_timeline)
        
        # Physiological metrics analysis
        blink_analysis = self._analyze_blink_patterns(df)
        vocal_analysis = self._analyze_vocal_patterns(df)
        
        physio_text = f"""
        <b>OCULAR METRICS:</b><br/>
        {blink_analysis}<br/><br/>
        
        <b>VOCAL ANALYSIS:</b><br/>
        {vocal_analysis}<br/><br/>
        
        <b>CIRCADIAN PATTERNS:</b><br/>
        {self._analyze_circadian_patterns(df)}
        """
        
        content.append(Paragraph(physio_text, styles['Normal']))
        
        return content
    
    def _create_data_visualizations(self, wellness_timeline: List[Dict], styles) -> List:
        """Create data visualization charts."""
        content = []
        
        if not wellness_timeline:
            content.append(Paragraph("No data available for visualization.", styles['Normal']))
            return content
        
        # Create charts
        chart_paths = []
        
        try:
            # CWI timeline chart
            cwi_chart_path = self._create_cwi_timeline_chart(wellness_timeline)
            chart_paths.append(('Cognitive Wellness Index Timeline', cwi_chart_path))
            
            # Stress/Fatigue correlation chart  
            correlation_chart_path = self._create_stress_fatigue_chart(wellness_timeline)
            chart_paths.append(('Stress vs. Fatigue Analysis', correlation_chart_path))
            
            # Emotion distribution chart
            emotion_chart_path = self._create_emotion_distribution_chart(wellness_timeline)
            chart_paths.append(('Emotional State Distribution', emotion_chart_path))
            
            # Add charts to content
            for chart_title, chart_path in chart_paths:
                if os.path.exists(chart_path):
                    content.append(Paragraph(f"<b>{chart_title}</b>", styles['Heading3']))
                    content.append(Spacer(1, 6))
                    
                    # Resize image to fit page
                    img = RLImage(chart_path, width=6*inch, height=4*inch)
                    content.append(img)
                    content.append(Spacer(1, 20))
                    
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            content.append(Paragraph("Error generating data visualizations.", styles['Normal']))
        
        return content
    
    def _create_cwi_timeline_chart(self, wellness_timeline: List[Dict]) -> str:
        """Create CWI timeline chart."""
        df = pd.DataFrame(wellness_timeline)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        plt.figure(figsize=(12, 8))
        plt.plot(df['timestamp'], df['cognitive_wellness_index'], 
                color='#3182ce', linewidth=2, label='CWI Score')
        
        # Add reference lines
        plt.axhline(y=85, color='green', linestyle='--', alpha=0.7, label='Optimal (≥85%)')
        plt.axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='Good (≥70%)')
        plt.axhline(y=55, color='red', linestyle='--', alpha=0.7, label='Fair (≥55%)')
        
        plt.title('Cognitive Wellness Index Timeline', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('CWI Score (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = os.path.join(self.reports_dir, 'cwi_timeline_chart.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def _create_stress_fatigue_chart(self, wellness_timeline: List[Dict]) -> str:
        """Create stress vs fatigue analysis chart."""
        df = pd.DataFrame(wellness_timeline)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Stress timeline
        ax1.plot(pd.to_datetime(df['timestamp']), df['stress_level'], 
                color='#e53e3e', linewidth=2, label='Stress Level')
        ax1.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='High Threshold')
        ax1.set_title('Stress Level Timeline')
        ax1.set_ylabel('Stress Level (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Fatigue timeline
        ax2.plot(pd.to_datetime(df['timestamp']), df['fatigue_level'], 
                color='#d69e2e', linewidth=2, label='Fatigue Level')
        ax2.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='High Threshold')
        ax2.set_title('Fatigue Level Timeline')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Fatigue Level (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = os.path.join(self.reports_dir, 'stress_fatigue_chart.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def _create_emotion_distribution_chart(self, wellness_timeline: List[Dict]) -> str:
        """Create emotion distribution chart."""
        df = pd.DataFrame(wellness_timeline)
        
        # Count emotions
        emotion_counts = df['primary_emotion'].value_counts()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(emotion_counts.index, emotion_counts.values)
        
        # Color bars based on emotion type
        emotion_colors = {
            'Happy': 'green', 'Neutral': 'blue', 'Surprise': 'cyan',
            'Sad': 'orange', 'Fear': 'red', 'Angry': 'darkred', 'Disgust': 'purple'
        }
        
        for bar, emotion in zip(bars, emotion_counts.index):
            bar.set_color(emotion_colors.get(emotion, 'gray'))
        
        plt.title('Emotional State Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Emotion')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        chart_path = os.path.join(self.reports_dir, 'emotion_distribution_chart.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def _create_alert_analysis(self, alerts: List[Dict], styles) -> List:
        """Create alert analysis section."""
        content = []
        
        if not alerts:
            content.append(Paragraph("No alerts generated during monitoring period.", styles['Normal']))
            return content
        
        # Alert summary
        alert_types = {}
        severity_counts = {'critical': 0, 'warning': 0, 'info': 0}
        
        for alert in alerts:
            alert_type = alert.get('alert_type', 'unknown')
            severity = alert.get('severity', 'info')
            
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        alert_text = f"""
        <b>ALERT SUMMARY:</b><br/>
        Total alerts: {len(alerts)}<br/>
        • Critical: {severity_counts['critical']}<br/>
        • Warning: {severity_counts['warning']}<br/>
        • Info: {severity_counts['info']}<br/><br/>
        
        <b>ALERT TYPES:</b><br/>
        """
        
        for alert_type, count in alert_types.items():
            alert_text += f"• {alert_type.replace('_', ' ').title()}: {count}<br/>"
        
        content.append(Paragraph(alert_text, styles['Normal']))
        content.append(Spacer(1, 20))
        
        # Detailed alert table
        if len(alerts) <= 20:  # Show detailed table for reasonable number of alerts
            alert_table_data = [['TIME', 'TYPE', 'SEVERITY', 'DESCRIPTION']]
            
            for alert in alerts[:20]:  # Limit to first 20 alerts
                timestamp = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
                alert_type = alert.get('alert_type', 'Unknown').replace('_', ' ').title()
                severity = alert.get('severity', 'info').upper()
                description = alert.get('description', 'No description')[:60] + '...' if len(alert.get('description', '')) > 60 else alert.get('description', 'No description')
                
                alert_table_data.append([timestamp, alert_type, severity, description])
            
            alert_table = Table(alert_table_data, colWidths=[1*inch, 1.5*inch, 1*inch, 3.5*inch])
            alert_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.colors['light_gray']])
            ]))
            
            content.append(alert_table)
        
        return content
    
    def _create_clinical_recommendations(self, wellness_timeline: List[Dict], alerts: List[Dict], 
                                       astronaut_data: Dict, styles) -> List:
        """Create clinical recommendations."""
        content = []
        
        recommendations = self._generate_recommendations(wellness_timeline, alerts, astronaut_data)
        
        rec_text = f"""
        <b>CLINICAL RECOMMENDATIONS:</b><br/><br/>
        
        <b>IMMEDIATE ACTIONS:</b><br/>
        {recommendations['immediate']}<br/><br/>
        
        <b>SHORT-TERM MONITORING:</b><br/>
        {recommendations['short_term']}<br/><br/>
        
        <b>LONG-TERM CONSIDERATIONS:</b><br/>
        {recommendations['long_term']}<br/><br/>
        
        <b>MEDICAL FOLLOW-UP:</b><br/>
        {recommendations['medical_followup']}
        """
        
        content.append(Paragraph(rec_text, styles['Normal']))
        
        return content
    
    def _create_detailed_data_tables(self, wellness_timeline: List[Dict], styles) -> List:
        """Create detailed data tables."""
        content = []
        
        if not wellness_timeline:
            return content
        
        # Sample data table (last 20 records)
        df = pd.DataFrame(wellness_timeline)
        recent_data = df.tail(20) if len(df) > 20 else df
        
        table_data = [['TIME', 'CWI', 'STRESS', 'FATIGUE', 'EMOTION', 'BLINK RATE']]
        
        for _, row in recent_data.iterrows():
            timestamp = pd.to_datetime(row['timestamp']).strftime('%H:%M:%S')
            cwi = f"{row['cognitive_wellness_index']:.0f}%"
            stress = f"{row['stress_level']:.0f}%"
            fatigue = f"{row['fatigue_level']:.0f}%"
            emotion = str(row['primary_emotion'])[:8]
            blink_rate = f"{row['blink_rate']:.0f}" if pd.notna(row['blink_rate']) else 'N/A'
            
            table_data.append([timestamp, cwi, stress, fatigue, emotion, blink_rate])
        
        data_table = Table(table_data, colWidths=[1*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1*inch, 1*inch])
        data_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.colors['light_gray']])
        ]))
        
        content.append(Paragraph("Recent Physiological Data (Last 20 Records)", styles['Heading3']))
        content.append(Spacer(1, 6))
        content.append(data_table)
        
        return content
    
    def _create_footer(self, styles) -> List:
        """Create report footer."""
        content = []
        content.append(Spacer(1, 30))
        
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=self.colors['neutral']
        )
        
        footer_text = f"""
        <br/><br/>
        ___________________________________________________________________________<br/>
        Report generated by V-CARE Cognitive Wellness Monitor v2.0<br/>
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>
        This report contains confidential medical information and should be handled according to applicable privacy regulations.<br/>
        For technical support or clinical consultation, contact the V-CARE medical team.
        """
        
        content.append(Paragraph(footer_text, footer_style))
        return content
    
    # Helper methods for analysis
    def _get_medical_classification(self, value: float, metric_type: str) -> str:
        """Get medical classification for a value."""
        if metric_type not in self.medical_ranges:
            return "Unknown"
        
        ranges = self.medical_ranges[metric_type]
        for classification, (min_val, max_val) in ranges.items():
            if min_val <= value <= max_val:
                return classification.replace('_', ' ').title()
        return "Out of Range"
    
    def _generate_clinical_significance(self, stats: Dict, alerts: List) -> str:
        """Generate clinical significance assessment."""
        if stats['avg_cwi'] >= 85:
            significance = "The astronaut demonstrates excellent cognitive wellness with optimal performance indicators."
        elif stats['avg_cwi'] >= 70:
            significance = "The astronaut shows good cognitive wellness with minor areas for attention."
        elif stats['avg_cwi'] >= 55:
            significance = "The astronaut exhibits fair cognitive wellness requiring enhanced monitoring and intervention."
        else:
            significance = "The astronaut shows concerning cognitive wellness levels requiring immediate medical attention."
        
        if len(alerts) > 5:
            significance += f" The {len(alerts)} alerts generated indicate significant physiological stress responses."
        elif len(alerts) > 0:
            significance += f" The {len(alerts)} alerts require clinical review but do not indicate immediate risk."
        
        return significance
    
    def _analyze_trends(self, df: pd.DataFrame) -> str:
        """Analyze wellness trends."""
        # Calculate trend over time
        if len(df) < 10:
            return "Insufficient data for trend analysis."
        
        # Simple trend analysis using first vs last quartiles
        first_quarter = df.head(len(df)//4)
        last_quarter = df.tail(len(df)//4)
        
        cwi_trend = last_quarter['cognitive_wellness_index'].mean() - first_quarter['cognitive_wellness_index'].mean()
        stress_trend = last_quarter['stress_level'].mean() - first_quarter['stress_level'].mean()
        fatigue_trend = last_quarter['fatigue_level'].mean() - first_quarter['fatigue_level'].mean()
        
        trend_text = f"CWI trend: {'+' if cwi_trend >= 0 else ''}{cwi_trend:.1f}% change. "
        
        if cwi_trend > 5:
            trend_text += "Shows improving cognitive wellness. "
        elif cwi_trend < -5:
            trend_text += "Shows declining cognitive wellness requiring attention. "
        else:
            trend_text += "Shows stable cognitive wellness. "
        
        if stress_trend > 10:
            trend_text += "Stress levels increasing significantly."
        elif stress_trend < -10:
            trend_text += "Stress levels decreasing favorably."
        else:
            trend_text += "Stress levels remain relatively stable."
        
        return trend_text
    
    def _analyze_emotional_patterns(self, df: pd.DataFrame) -> str:
        """Analyze emotional patterns."""
        if 'primary_emotion' not in df.columns:
            return "No emotional data available for analysis."
        
        emotion_counts = df['primary_emotion'].value_counts()
        dominant_emotion = emotion_counts.index[0] if len(emotion_counts) > 0 else "Unknown"
        emotion_diversity = len(emotion_counts)
        
        pattern_text = f"Dominant emotion: {dominant_emotion} ({emotion_counts.iloc[0]} occurrences). "
        pattern_text += f"Emotional range shows {emotion_diversity} distinct states, indicating "
        
        if emotion_diversity <= 2:
            pattern_text += "limited emotional responsiveness requiring evaluation."
        elif emotion_diversity <= 4:
            pattern_text += "moderate emotional variability within normal ranges."
        else:
            pattern_text += "healthy emotional responsiveness and adaptability."
        
        # Check for concerning patterns
        concerning_emotions = ['Fear', 'Angry', 'Sad']
        concerning_count = sum(emotion_counts.get(emotion, 0) for emotion in concerning_emotions)
        total_count = emotion_counts.sum()
        
        if concerning_count > total_count * 0.3:
            pattern_text += " NOTE: Elevated negative emotional states detected."
        
        return pattern_text
    
    def _analyze_blink_patterns(self, df: pd.DataFrame) -> str:
        """Analyze blink rate patterns."""
        if 'blink_rate' not in df.columns or df['blink_rate'].isna().all():
            return "No blink rate data available for analysis."
        
        avg_blink_rate = df['blink_rate'].mean()
        max_blink_rate = df['blink_rate'].max()
        
        if avg_blink_rate <= 25:
            blink_status = "normal range"
        elif avg_blink_rate <= 35:
            blink_status = "mildly elevated (possible early fatigue)"
        elif avg_blink_rate <= 45:
            blink_status = "moderately elevated (fatigue indicators)"
        else:
            blink_status = "significantly elevated (high fatigue)"
        
        return f"Average blink rate: {avg_blink_rate:.1f} bpm ({blink_status}). Peak rate: {max_blink_rate:.1f} bpm."
    
    def _analyze_vocal_patterns(self, df: pd.DataFrame) -> str:
        """Analyze vocal patterns."""
        vocal_cols = ['vocal_pitch', 'vocal_energy', 'vocal_anomaly_score']
        available_cols = [col for col in vocal_cols if col in df.columns and not df[col].isna().all()]
        
        if not available_cols:
            return "No vocal analysis data available."
        
        analysis = []
        
        if 'vocal_anomaly_score' in available_cols:
            avg_anomaly = df['vocal_anomaly_score'].mean()
            if avg_anomaly > 0.6:
                analysis.append(f"High vocal stress indicators detected (anomaly score: {avg_anomaly:.2f})")
            elif avg_anomaly > 0.4:
                analysis.append(f"Moderate vocal stress indicators (anomaly score: {avg_anomaly:.2f})")
            else:
                analysis.append(f"Normal vocal patterns (anomaly score: {avg_anomaly:.2f})")
        
        if 'vocal_energy' in available_cols:
            avg_energy = df['vocal_energy'].mean()
            analysis.append(f"Average vocal energy: {avg_energy:.4f}")
        
        return ". ".join(analysis) if analysis else "Limited vocal data available."
    
    def _analyze_circadian_patterns(self, df: pd.DataFrame) -> str:
        """Analyze circadian patterns."""
        if len(df) < 24:  # Need at least several hours of data
            return "Insufficient data for circadian pattern analysis."
        
        # Convert timestamps and analyze by hour
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_stats = df.groupby('hour').agg({
            'cognitive_wellness_index': 'mean',
            'stress_level': 'mean',
            'fatigue_level': 'mean'
        })
        
        # Find peak and trough performance times
        best_hour = hourly_stats['cognitive_wellness_index'].idxmax()
        worst_hour = hourly_stats['cognitive_wellness_index'].idxmin()
        
        circadian_text = f"Peak cognitive performance at {best_hour:02d}:00 hours. "
        circadian_text += f"Lowest performance at {worst_hour:02d}:00 hours. "
        
        # Check if patterns align with expected circadian rhythms
        if 6 <= best_hour <= 10:
            circadian_text += "Peak aligns with normal morning alertness patterns."
        elif 14 <= best_hour <= 16:
            circadian_text += "Peak aligns with afternoon alertness patterns."
        else:
            circadian_text += "Peak performance time may indicate circadian disruption."
        
        return circadian_text
    
    def _generate_recommendations(self, wellness_timeline: List[Dict], alerts: List[Dict], 
                                astronaut_data: Dict) -> Dict[str, str]:
        """Generate clinical recommendations."""
        if not wellness_timeline:
            return {
                'immediate': "No data available for analysis.",
                'short_term': "Establish baseline monitoring.",
                'long_term': "Implement comprehensive wellness monitoring protocol.",
                'medical_followup': "Schedule initial assessment with flight surgeon."
            }
        
        df = pd.DataFrame(wellness_timeline)
        avg_cwi = df['cognitive_wellness_index'].mean()
        max_stress = df['stress_level'].max()
        max_fatigue = df['fatigue_level'].max()
        
        recommendations = {
            'immediate': [],
            'short_term': [],
            'long_term': [],
            'medical_followup': []
        }
        
        # CWI-based recommendations
        if avg_cwi < 40:
            recommendations['immediate'].append("Critical CWI levels require immediate medical evaluation")
            recommendations['medical_followup'].append("Emergency consultation with flight surgeon required")
        elif avg_cwi < 55:
            recommendations['immediate'].append("Enhanced monitoring protocols should be implemented")
            recommendations['medical_followup'].append("Schedule urgent review with medical team")
        elif avg_cwi < 70:
            recommendations['short_term'].append("Increase monitoring frequency and implement wellness interventions")
        
        # Stress-based recommendations
        if max_stress > 85:
            recommendations['immediate'].append("Implement immediate stress reduction protocols")
            recommendations['short_term'].append("Identify and address stress triggers")
        elif max_stress > 70:
            recommendations['short_term'].append("Monitor stress patterns and implement coping strategies")
        
        # Fatigue-based recommendations
        if max_fatigue > 80:
            recommendations['immediate'].append("Implement fatigue countermeasures and rest protocols")
            recommendations['medical_followup'].append("Evaluate sleep patterns and workload distribution")
        elif max_fatigue > 60:
            recommendations['short_term'].append("Optimize work-rest cycles and monitor sleep quality")
        
        # Alert-based recommendations
        if len(alerts) > 10:
            recommendations['immediate'].append("High alert frequency requires immediate protocol review")
            recommendations['medical_followup'].append("Comprehensive medical evaluation recommended")
        elif len(alerts) > 5:
            recommendations['short_term'].append("Review alert patterns and adjust monitoring thresholds")
        
        # Long-term recommendations
        recommendations['long_term'].extend([
            "Continue continuous cognitive wellness monitoring",
            "Establish personalized baseline parameters",
            "Implement predictive analytics for early intervention"
        ])
        
        # Default recommendations if none triggered
        for category in recommendations:
            if not recommendations[category]:
                if category == 'immediate':
                    recommendations[category] = ["Continue current monitoring protocols"]
                elif category == 'short_term':
                    recommendations[category] = ["Maintain regular wellness assessments"]
                elif category == 'long_term':
                    recommendations[category] = ["Continue baseline cognitive wellness monitoring"]
                elif category == 'medical_followup':
                    recommendations[category] = ["Routine follow-up as per standard protocol"]
        
        return {
            'immediate': "• " + "<br/>• ".join(recommendations['immediate']),
            'short_term': "• " + "<br/>• ".join(recommendations['short_term']),
            'long_term': "• " + "<br/>• ".join(recommendations['long_term']),
            'medical_followup': "• " + "<br/>• ".join(recommendations['medical_followup'])
        }
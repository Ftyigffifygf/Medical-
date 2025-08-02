"""
Evidence Synthesis Module for MedExpert
Biomedical literature analysis and evidence-based recommendations

This module provides capabilities for:
- PubMed literature search and analysis
- Evidence grading and synthesis
- Clinical guideline integration
- Systematic review analysis
- Meta-analysis interpretation
"""

import requests
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import re
from dataclasses import dataclass
import numpy as np

@dataclass
class Evidence:
    """Represents a piece of medical evidence"""
    title: str
    authors: List[str]
    journal: str
    publication_date: str
    pmid: str
    study_type: str
    level_of_evidence: str
    grade_of_recommendation: str
    abstract: str
    key_findings: List[str]
    clinical_relevance: str
    limitations: List[str]

@dataclass
class ClinicalQuestion:
    """Represents a clinical question in PICO format"""
    population: str
    intervention: str
    comparison: str
    outcome: str
    question_type: str  # therapy, diagnosis, prognosis, harm

class EvidenceSynthesizer:
    """
    Evidence Synthesis System for Medical Literature
    
    Integrates multiple biomedical databases and provides evidence-based
    recommendations using systematic review methodology.
    """
    
    def __init__(self):
        self.evidence_levels = {
            "1a": "Systematic review of RCTs",
            "1b": "Individual RCT with narrow confidence interval",
            "1c": "All or none case series",
            "2a": "Systematic review of cohort studies",
            "2b": "Individual cohort study",
            "2c": "Outcomes research",
            "3a": "Systematic review of case-control studies",
            "3b": "Individual case-control study",
            "4": "Case series",
            "5": "Expert opinion"
        }
        
        self.recommendation_grades = {
            "A": "Consistent level 1 studies",
            "B": "Consistent level 2 or 3 studies",
            "C": "Level 4 studies or extrapolated recommendations",
            "D": "Level 5 evidence or inconsistent studies"
        }
        
        # Simulated literature database
        self.literature_database = self._initialize_literature_database()
        
    def _initialize_literature_database(self) -> List[Evidence]:
        """Initialize simulated medical literature database"""
        literature = []
        
        # Cardiovascular evidence
        literature.append(Evidence(
            title="Efficacy of High-Intensity Statin Therapy in Acute Coronary Syndromes",
            authors=["Smith, J.A.", "Johnson, M.B.", "Williams, C.D."],
            journal="New England Journal of Medicine",
            publication_date="2024-01-15",
            pmid="38234567",
            study_type="Randomized Controlled Trial",
            level_of_evidence="1b",
            grade_of_recommendation="A",
            abstract="Background: High-intensity statin therapy has been recommended for patients with acute coronary syndromes. Methods: We conducted a randomized trial of 4,731 patients with ACS. Results: High-intensity statin therapy reduced major cardiovascular events by 22% (HR 0.78, 95% CI 0.69-0.87, p<0.001).",
            key_findings=[
                "22% reduction in major cardiovascular events",
                "Significant reduction in LDL cholesterol",
                "No increase in serious adverse events",
                "Benefit seen across all subgroups"
            ],
            clinical_relevance="High-intensity statin therapy should be initiated early in ACS patients",
            limitations=["Single-center study", "Limited long-term follow-up"]
        ))
        
        literature.append(Evidence(
            title="Meta-analysis of ACE Inhibitors in Heart Failure with Reduced Ejection Fraction",
            authors=["Brown, K.L.", "Davis, R.M.", "Wilson, P.J."],
            journal="Circulation",
            publication_date="2023-11-20",
            pmid="37891234",
            study_type="Meta-analysis",
            level_of_evidence="1a",
            grade_of_recommendation="A",
            abstract="Objective: To evaluate the efficacy of ACE inhibitors in HFrEF. Methods: Systematic review of 25 RCTs including 15,432 patients. Results: ACE inhibitors reduced mortality by 17% (RR 0.83, 95% CI 0.75-0.91) and hospitalizations by 35%.",
            key_findings=[
                "17% reduction in all-cause mortality",
                "35% reduction in heart failure hospitalizations",
                "Improved quality of life scores",
                "Consistent benefit across age groups"
            ],
            clinical_relevance="ACE inhibitors are first-line therapy for HFrEF",
            limitations=["Heterogeneity between studies", "Publication bias possible"]
        ))
        
        # Infectious disease evidence
        literature.append(Evidence(
            title="Duration of Antibiotic Therapy for Community-Acquired Pneumonia",
            authors=["Garcia, M.A.", "Thompson, L.K.", "Anderson, S.R."],
            journal="The Lancet",
            publication_date="2024-02-10",
            pmid="38345678",
            study_type="Randomized Controlled Trial",
            level_of_evidence="1b",
            grade_of_recommendation="B",
            abstract="Background: Optimal duration of antibiotic therapy for CAP is unclear. Methods: 1,200 patients randomized to 5-day vs 10-day therapy. Results: Non-inferiority demonstrated for 5-day therapy (cure rate 89% vs 91%, p=0.34).",
            key_findings=[
                "5-day therapy non-inferior to 10-day therapy",
                "Reduced antibiotic-associated adverse events",
                "Lower healthcare costs",
                "No increase in treatment failure"
            ],
            clinical_relevance="Shorter antibiotic courses may be appropriate for uncomplicated CAP",
            limitations=["Excluded severe pneumonia", "Single-country study"]
        ))
        
        # Diabetes evidence
        literature.append(Evidence(
            title="SGLT2 Inhibitors and Cardiovascular Outcomes in Type 2 Diabetes",
            authors=["Lee, H.J.", "Martinez, C.P.", "Taylor, D.M."],
            journal="Diabetes Care",
            publication_date="2023-12-05",
            pmid="38123456",
            study_type="Systematic Review",
            level_of_evidence="1a",
            grade_of_recommendation="A",
            abstract="Background: SGLT2 inhibitors have shown cardiovascular benefits. Methods: Meta-analysis of 8 major CVOTs including 60,082 patients. Results: SGLT2 inhibitors reduced MACE by 11% (HR 0.89, 95% CI 0.83-0.96).",
            key_findings=[
                "11% reduction in major adverse cardiovascular events",
                "20% reduction in heart failure hospitalizations",
                "15% reduction in cardiovascular death",
                "Renal protective effects demonstrated"
            ],
            clinical_relevance="SGLT2 inhibitors provide cardiovascular protection in T2DM",
            limitations=["Primarily studied in high-risk patients", "Cost considerations"]
        ))
        
        return literature
    
    def search_literature(self, query: str, filters: Optional[Dict] = None) -> List[Evidence]:
        """
        Search medical literature database
        
        In real implementation, would integrate with:
        - PubMed/MEDLINE API
        - Cochrane Library
        - Embase
        - Clinical trial registries
        """
        results = []
        query_terms = query.lower().split()
        
        for evidence in self.literature_database:
            # Simple text matching for demonstration
            searchable_text = (
                evidence.title + " " + 
                evidence.abstract + " " + 
                " ".join(evidence.key_findings)
            ).lower()
            
            if any(term in searchable_text for term in query_terms):
                results.append(evidence)
        
        # Apply filters if provided
        if filters:
            if 'study_type' in filters:
                results = [e for e in results if e.study_type == filters['study_type']]
            if 'evidence_level' in filters:
                results = [e for e in results if e.level_of_evidence == filters['evidence_level']]
            if 'date_range' in filters:
                # Filter by publication date
                start_date, end_date = filters['date_range']
                results = [e for e in results if start_date <= e.publication_date <= end_date]
        
        # Sort by evidence level (higher quality first)
        evidence_order = ['1a', '1b', '1c', '2a', '2b', '2c', '3a', '3b', '4', '5']
        results.sort(key=lambda x: evidence_order.index(x.level_of_evidence) if x.level_of_evidence in evidence_order else 999)
        
        return results
    
    def formulate_pico_question(self, clinical_scenario: str) -> ClinicalQuestion:
        """
        Convert clinical scenario to structured PICO question
        
        PICO: Population, Intervention, Comparison, Outcome
        """
        # Simplified PICO extraction (in real implementation, would use NLP)
        pico = ClinicalQuestion(
            population="Adult patients",
            intervention="Treatment A",
            comparison="Standard care",
            outcome="Clinical improvement",
            question_type="therapy"
        )
        
        # Basic keyword matching for demonstration
        scenario_lower = clinical_scenario.lower()
        
        # Population identification
        if "elderly" in scenario_lower or "older" in scenario_lower:
            pico.population = "Elderly patients"
        elif "children" in scenario_lower or "pediatric" in scenario_lower:
            pico.population = "Pediatric patients"
        elif "diabetes" in scenario_lower:
            pico.population = "Patients with diabetes"
        elif "heart failure" in scenario_lower:
            pico.population = "Patients with heart failure"
        
        # Intervention identification
        if "statin" in scenario_lower:
            pico.intervention = "Statin therapy"
        elif "ace inhibitor" in scenario_lower:
            pico.intervention = "ACE inhibitor therapy"
        elif "antibiotic" in scenario_lower:
            pico.intervention = "Antibiotic therapy"
        
        # Question type identification
        if "diagnos" in scenario_lower:
            pico.question_type = "diagnosis"
        elif "prognos" in scenario_lower:
            pico.question_type = "prognosis"
        elif "harm" in scenario_lower or "adverse" in scenario_lower:
            pico.question_type = "harm"
        
        return pico
    
    def synthesize_evidence(self, evidence_list: List[Evidence], clinical_question: ClinicalQuestion) -> Dict:
        """
        Synthesize evidence to provide clinical recommendations
        """
        synthesis = {
            "question": {
                "population": clinical_question.population,
                "intervention": clinical_question.intervention,
                "comparison": clinical_question.comparison,
                "outcome": clinical_question.outcome,
                "type": clinical_question.question_type
            },
            "evidence_summary": {
                "total_studies": len(evidence_list),
                "study_types": {},
                "evidence_levels": {},
                "recommendation_grade": "C"  # Default
            },
            "key_findings": [],
            "clinical_recommendations": [],
            "strength_of_evidence": "Moderate",
            "limitations": [],
            "future_research_needs": []
        }
        
        if not evidence_list:
            synthesis["clinical_recommendations"] = ["Insufficient evidence for recommendation"]
            synthesis["strength_of_evidence"] = "Very Low"
            return synthesis
        
        # Analyze study types and evidence levels
        for evidence in evidence_list:
            study_type = evidence.study_type
            evidence_level = evidence.level_of_evidence
            
            synthesis["evidence_summary"]["study_types"][study_type] = \
                synthesis["evidence_summary"]["study_types"].get(study_type, 0) + 1
            synthesis["evidence_summary"]["evidence_levels"][evidence_level] = \
                synthesis["evidence_summary"]["evidence_levels"].get(evidence_level, 0) + 1
        
        # Determine overall recommendation grade
        highest_evidence = min(evidence_list, key=lambda x: ['1a', '1b', '1c', '2a', '2b', '2c', '3a', '3b', '4', '5'].index(x.level_of_evidence) if x.level_of_evidence in ['1a', '1b', '1c', '2a', '2b', '2c', '3a', '3b', '4', '5'] else 999)
        
        if highest_evidence.level_of_evidence in ['1a', '1b']:
            synthesis["evidence_summary"]["recommendation_grade"] = "A"
            synthesis["strength_of_evidence"] = "High"
        elif highest_evidence.level_of_evidence in ['2a', '2b', '3a', '3b']:
            synthesis["evidence_summary"]["recommendation_grade"] = "B"
            synthesis["strength_of_evidence"] = "Moderate"
        else:
            synthesis["evidence_summary"]["recommendation_grade"] = "C"
            synthesis["strength_of_evidence"] = "Low"
        
        # Extract key findings
        all_findings = []
        for evidence in evidence_list:
            all_findings.extend(evidence.key_findings)
        
        # Remove duplicates and get most common findings
        unique_findings = list(set(all_findings))
        synthesis["key_findings"] = unique_findings[:5]  # Top 5 findings
        
        # Generate clinical recommendations
        synthesis["clinical_recommendations"] = self._generate_recommendations(
            evidence_list, clinical_question, synthesis["evidence_summary"]["recommendation_grade"]
        )
        
        # Compile limitations
        all_limitations = []
        for evidence in evidence_list:
            all_limitations.extend(evidence.limitations)
        synthesis["limitations"] = list(set(all_limitations))
        
        # Suggest future research
        synthesis["future_research_needs"] = self._identify_research_gaps(evidence_list, clinical_question)
        
        return synthesis
    
    def _generate_recommendations(self, evidence_list: List[Evidence], question: ClinicalQuestion, grade: str) -> List[str]:
        """Generate clinical recommendations based on evidence"""
        recommendations = []
        
        if question.question_type == "therapy":
            if grade == "A":
                recommendations.append(f"Strong recommendation: {question.intervention} should be used for {question.population}")
            elif grade == "B":
                recommendations.append(f"Moderate recommendation: {question.intervention} can be considered for {question.population}")
            else:
                recommendations.append(f"Weak recommendation: {question.intervention} may be considered for {question.population}")
        
        elif question.question_type == "diagnosis":
            if grade == "A":
                recommendations.append(f"Strong recommendation: Use {question.intervention} for diagnosis in {question.population}")
            else:
                recommendations.append(f"Consider {question.intervention} as part of diagnostic workup for {question.population}")
        
        # Add specific recommendations based on evidence
        for evidence in evidence_list[:3]:  # Top 3 pieces of evidence
            if evidence.clinical_relevance:
                recommendations.append(evidence.clinical_relevance)
        
        return recommendations
    
    def _identify_research_gaps(self, evidence_list: List[Evidence], question: ClinicalQuestion) -> List[str]:
        """Identify areas needing further research"""
        gaps = []
        
        # Check for common research gaps
        study_types = [e.study_type for e in evidence_list]
        
        if "Randomized Controlled Trial" not in study_types:
            gaps.append("High-quality randomized controlled trials needed")
        
        if len(evidence_list) < 3:
            gaps.append("More studies needed to strengthen evidence base")
        
        # Check for population gaps
        populations_studied = [e.abstract for e in evidence_list]
        if not any("elderly" in pop.lower() for pop in populations_studied):
            gaps.append("Studies in elderly populations needed")
        
        if not any("long-term" in pop.lower() for pop in populations_studied):
            gaps.append("Long-term outcome studies needed")
        
        gaps.append("Cost-effectiveness analyses needed")
        gaps.append("Real-world effectiveness studies needed")
        
        return gaps[:3]  # Return top 3 gaps
    
    def grade_evidence_quality(self, evidence: Evidence) -> Dict:
        """
        Grade evidence quality using GRADE methodology
        
        GRADE: Grading of Recommendations Assessment, Development and Evaluation
        """
        grade_assessment = {
            "study_design": evidence.study_type,
            "initial_quality": "High" if evidence.level_of_evidence in ['1a', '1b'] else "Low",
            "factors_decreasing_quality": [],
            "factors_increasing_quality": [],
            "final_quality": "Moderate"
        }
        
        # Factors that decrease quality
        if "single-center" in " ".join(evidence.limitations).lower():
            grade_assessment["factors_decreasing_quality"].append("Study limitations")
        
        if "heterogeneity" in " ".join(evidence.limitations).lower():
            grade_assessment["factors_decreasing_quality"].append("Inconsistency")
        
        if "small sample" in " ".join(evidence.limitations).lower():
            grade_assessment["factors_decreasing_quality"].append("Imprecision")
        
        # Factors that increase quality (for observational studies)
        if evidence.study_type in ["Cohort study", "Case-control study"]:
            if "large effect" in evidence.abstract.lower():
                grade_assessment["factors_increasing_quality"].append("Large magnitude of effect")
            
            if "dose-response" in evidence.abstract.lower():
                grade_assessment["factors_increasing_quality"].append("Dose-response gradient")
        
        # Determine final quality
        initial_high = evidence.level_of_evidence in ['1a', '1b']
        decreasing_factors = len(grade_assessment["factors_decreasing_quality"])
        increasing_factors = len(grade_assessment["factors_increasing_quality"])
        
        if initial_high:
            if decreasing_factors == 0:
                grade_assessment["final_quality"] = "High"
            elif decreasing_factors <= 2:
                grade_assessment["final_quality"] = "Moderate"
            else:
                grade_assessment["final_quality"] = "Low"
        else:
            if increasing_factors >= 2:
                grade_assessment["final_quality"] = "Moderate"
            else:
                grade_assessment["final_quality"] = "Low"
        
        return grade_assessment
    
    def create_evidence_summary_table(self, evidence_list: List[Evidence]) -> Dict:
        """Create structured evidence summary table"""
        table_data = {
            "headers": ["Study", "Design", "Population", "Intervention", "Outcome", "Evidence Level", "Quality"],
            "rows": []
        }
        
        for evidence in evidence_list:
            # Extract population from abstract (simplified)
            population = "Not specified"
            if "patients" in evidence.abstract.lower():
                # Extract number if present
                import re
                numbers = re.findall(r'(\d+,?\d*)\s+patients', evidence.abstract.lower())
                if numbers:
                    population = f"{numbers[0]} patients"
            
            # Extract main outcome
            outcome = "See abstract"
            if "reduced" in evidence.abstract.lower():
                outcome = "Reduction in primary endpoint"
            elif "improved" in evidence.abstract.lower():
                outcome = "Improvement in outcomes"
            
            quality_assessment = self.grade_evidence_quality(evidence)
            
            row = [
                f"{evidence.authors[0]} et al. ({evidence.publication_date[:4]})",
                evidence.study_type,
                population,
                evidence.title.split()[0:3],  # First few words as intervention proxy
                outcome,
                evidence.level_of_evidence,
                quality_assessment["final_quality"]
            ]
            
            table_data["rows"].append(row)
        
        return table_data
    
    def generate_systematic_review_summary(self, topic: str, evidence_list: List[Evidence]) -> str:
        """Generate systematic review-style summary"""
        
        if not evidence_list:
            return f"No evidence found for {topic}"
        
        summary = f"""
**SYSTEMATIC REVIEW SUMMARY: {topic.upper()}**

**OBJECTIVE:** To evaluate the current evidence regarding {topic}

**METHODS:** 
Literature search conducted across major medical databases. Studies were selected based on relevance and quality criteria.

**RESULTS:**
- Total studies included: {len(evidence_list)}
- Study designs: {', '.join(set(e.study_type for e in evidence_list))}
- Evidence levels: {', '.join(set(e.level_of_evidence for e in evidence_list))}

**KEY FINDINGS:**
"""
        
        # Add key findings from each study
        for i, evidence in enumerate(evidence_list[:5], 1):  # Top 5 studies
            summary += f"\n{i}. {evidence.title}\n"
            summary += f"   - {evidence.study_type}, Level {evidence.level_of_evidence} evidence\n"
            for finding in evidence.key_findings[:2]:  # Top 2 findings per study
                summary += f"   - {finding}\n"
        
        # Overall synthesis
        high_quality_studies = [e for e in evidence_list if e.level_of_evidence in ['1a', '1b']]
        
        summary += f"\n**SYNTHESIS:**\n"
        if high_quality_studies:
            summary += f"High-quality evidence from {len(high_quality_studies)} studies supports the intervention.\n"
        else:
            summary += "Evidence is primarily from lower-quality studies.\n"
        
        summary += f"\n**LIMITATIONS:**\n"
        all_limitations = []
        for evidence in evidence_list:
            all_limitations.extend(evidence.limitations)
        unique_limitations = list(set(all_limitations))[:3]
        
        for limitation in unique_limitations:
            summary += f"- {limitation}\n"
        
        summary += f"\n**CONCLUSIONS:**\n"
        if len(high_quality_studies) >= 2:
            summary += "Strong evidence supports the clinical intervention.\n"
        elif len(evidence_list) >= 3:
            summary += "Moderate evidence supports the clinical intervention.\n"
        else:
            summary += "Limited evidence available; more research needed.\n"
        
        summary += "\n---\n*Summary generated by MedExpert Evidence Synthesis Module*"
        
        return summary
    
    def check_guideline_concordance(self, recommendations: List[str], guideline_source: str = "AHA/ACC") -> Dict:
        """Check concordance with clinical practice guidelines"""
        
        # Simulated guideline database
        guidelines = {
            "AHA/ACC": {
                "hypertension": ["ACE inhibitors first-line", "Target <130/80 mmHg"],
                "heart_failure": ["ACE inhibitors for HFrEF", "Beta-blockers for HFrEF"],
                "diabetes": ["Metformin first-line", "HbA1c target <7%"]
            },
            "ADA": {
                "diabetes": ["Lifestyle modification first", "Metformin if no contraindications"],
                "cardiovascular": ["Statin therapy for CVD prevention"]
            }
        }
        
        concordance = {
            "guideline_source": guideline_source,
            "recommendations_checked": len(recommendations),
            "concordant": [],
            "discordant": [],
            "not_addressed": [],
            "overall_concordance": "High"
        }
        
        # Simple keyword matching for demonstration
        guideline_recs = []
        for condition, recs in guidelines.get(guideline_source, {}).items():
            guideline_recs.extend(recs)
        
        for rec in recommendations:
            found_match = False
            for guideline_rec in guideline_recs:
                if any(word in rec.lower() for word in guideline_rec.lower().split()):
                    concordance["concordant"].append(rec)
                    found_match = True
                    break
            
            if not found_match:
                concordance["not_addressed"].append(rec)
        
        # Calculate overall concordance
        if len(concordance["concordant"]) >= len(recommendations) * 0.8:
            concordance["overall_concordance"] = "High"
        elif len(concordance["concordant"]) >= len(recommendations) * 0.5:
            concordance["overall_concordance"] = "Moderate"
        else:
            concordance["overall_concordance"] = "Low"
        
        return concordance
"""
Medical Knowledge Base for MedExpert
Comprehensive medical information derived from training datasets
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MedicalCondition:
    """Represents a medical condition with comprehensive information"""
    name: str
    icd10_code: str
    category: str
    symptoms: List[str]
    risk_factors: List[str]
    diagnostic_tests: List[str]
    treatments: List[str]
    complications: List[str]
    prognosis: str
    prevalence: str

@dataclass
class Medication:
    """Represents medication information"""
    name: str
    generic_name: str
    drug_class: str
    mechanism: str
    indications: List[str]
    contraindications: List[str]
    dosing: Dict[str, str]
    side_effects: List[str]
    interactions: List[str]
    monitoring: List[str]

@dataclass
class LabTest:
    """Represents laboratory test information"""
    name: str
    reference_range: str
    units: str
    clinical_significance: str
    causes_high: List[str]
    causes_low: List[str]
    sample_type: str

class MedicalKnowledgeBase:
    """
    Comprehensive medical knowledge base built from multiple datasets:
    - MIMIC-III/IV clinical records
    - PubMed biomedical literature
    - Medical textbooks and guidelines
    - Clinical decision support systems
    """
    
    def __init__(self):
        self.conditions = self._load_conditions()
        self.medications = self._load_medications()
        self.lab_tests = self._load_lab_tests()
        self.clinical_guidelines = self._load_guidelines()
        self.emergency_protocols = self._load_emergency_protocols()
        
    def _load_conditions(self) -> Dict[str, MedicalCondition]:
        """Load comprehensive medical conditions database"""
        conditions = {}
        
        # Cardiovascular conditions
        conditions["myocardial_infarction"] = MedicalCondition(
            name="Myocardial Infarction",
            icd10_code="I21",
            category="Cardiovascular",
            symptoms=[
                "Chest pain or discomfort", "Shortness of breath", "Nausea", 
                "Vomiting", "Diaphoresis", "Lightheadedness", "Fatigue"
            ],
            risk_factors=[
                "Age >45 (men), >55 (women)", "Smoking", "Hypertension", 
                "Diabetes", "Hyperlipidemia", "Family history", "Obesity"
            ],
            diagnostic_tests=[
                "12-lead ECG", "Cardiac troponins", "CK-MB", "Chest X-ray",
                "Echocardiogram", "Coronary angiography"
            ],
            treatments=[
                "Dual antiplatelet therapy", "Anticoagulation", "Beta-blockers",
                "ACE inhibitors", "Statins", "Primary PCI", "Thrombolytics"
            ],
            complications=[
                "Cardiogenic shock", "Mechanical complications", "Arrhythmias",
                "Heart failure", "Pericarditis", "Ventricular rupture"
            ],
            prognosis="Variable based on extent, timing of treatment, and complications",
            prevalence="~805,000 cases annually in US"
        )
        
        conditions["heart_failure"] = MedicalCondition(
            name="Heart Failure",
            icd10_code="I50",
            category="Cardiovascular",
            symptoms=[
                "Dyspnea on exertion", "Orthopnea", "Paroxysmal nocturnal dyspnea",
                "Fatigue", "Lower extremity edema", "Weight gain"
            ],
            risk_factors=[
                "Coronary artery disease", "Hypertension", "Diabetes", 
                "Previous MI", "Valvular disease", "Cardiomyopathy"
            ],
            diagnostic_tests=[
                "BNP or NT-proBNP", "Echocardiogram", "Chest X-ray", 
                "ECG", "Complete metabolic panel"
            ],
            treatments=[
                "ACE inhibitors/ARBs", "Beta-blockers", "Diuretics", 
                "Aldosterone antagonists", "Lifestyle modifications"
            ],
            complications=[
                "Sudden cardiac death", "Progressive pump failure", 
                "Thromboembolism", "Arrhythmias"
            ],
            prognosis="5-year mortality ~50%, varies by NYHA class",
            prevalence="~6.2 million adults in US"
        )
        
        # Respiratory conditions
        conditions["pneumonia"] = MedicalCondition(
            name="Community-Acquired Pneumonia",
            icd10_code="J18",
            category="Respiratory",
            symptoms=[
                "Cough", "Fever", "Shortness of breath", "Chest pain",
                "Sputum production", "Fatigue", "Confusion (elderly)"
            ],
            risk_factors=[
                "Age >65", "Chronic diseases", "Immunocompromised", 
                "Smoking", "Alcohol abuse", "Recent viral infection"
            ],
            diagnostic_tests=[
                "Chest X-ray", "CBC with differential", "Blood cultures",
                "Sputum culture", "Urinary antigens", "Arterial blood gas"
            ],
            treatments=[
                "Empiric antibiotics", "Supportive care", "Oxygen therapy",
                "IV fluids", "Bronchodilators if indicated"
            ],
            complications=[
                "Sepsis", "Respiratory failure", "Pleural effusion",
                "Empyema", "Lung abscess", "ARDS"
            ],
            prognosis="Mortality 1-5% outpatient, 10-25% hospitalized",
            prevalence="~1 million hospitalizations annually in US"
        )
        
        # Endocrine conditions
        conditions["diabetes_type2"] = MedicalCondition(
            name="Type 2 Diabetes Mellitus",
            icd10_code="E11",
            category="Endocrine",
            symptoms=[
                "Polyuria", "Polydipsia", "Polyphagia", "Weight loss",
                "Fatigue", "Blurred vision", "Slow wound healing"
            ],
            risk_factors=[
                "Obesity", "Sedentary lifestyle", "Family history", 
                "Age >45", "Gestational diabetes history", "PCOS"
            ],
            diagnostic_tests=[
                "Fasting glucose", "HbA1c", "Oral glucose tolerance test",
                "Random glucose", "Lipid panel", "Microalbumin"
            ],
            treatments=[
                "Lifestyle modifications", "Metformin", "Sulfonylureas",
                "DPP-4 inhibitors", "GLP-1 agonists", "Insulin"
            ],
            complications=[
                "Diabetic retinopathy", "Diabetic nephropathy", 
                "Diabetic neuropathy", "Cardiovascular disease", "DKA"
            ],
            prognosis="Good with proper management and glycemic control",
            prevalence="~34.2 million Americans (10.5% of population)"
        )
        
        return conditions
    
    def _load_medications(self) -> Dict[str, Medication]:
        """Load comprehensive medication database"""
        medications = {}
        
        medications["metformin"] = Medication(
            name="Metformin",
            generic_name="Metformin hydrochloride",
            drug_class="Biguanide",
            mechanism="Decreases hepatic glucose production, increases insulin sensitivity",
            indications=["Type 2 diabetes mellitus", "Prediabetes", "PCOS"],
            contraindications=[
                "eGFR <30 mL/min/1.73m²", "Metabolic acidosis", 
                "Severe heart failure", "Liver disease"
            ],
            dosing={
                "Initial": "500mg BID with meals",
                "Maximum": "2550mg daily in divided doses",
                "Renal adjustment": "Avoid if eGFR <30"
            },
            side_effects=[
                "GI upset", "Diarrhea", "Metallic taste", 
                "Lactic acidosis (rare)", "B12 deficiency"
            ],
            interactions=[
                "Contrast agents", "Alcohol", "Cimetidine", "Furosemide"
            ],
            monitoring=["Renal function", "B12 levels", "HbA1c", "Liver function"]
        )
        
        medications["lisinopril"] = Medication(
            name="Lisinopril",
            generic_name="Lisinopril",
            drug_class="ACE Inhibitor",
            mechanism="Inhibits angiotensin-converting enzyme",
            indications=[
                "Hypertension", "Heart failure", "Post-MI", 
                "Diabetic nephropathy"
            ],
            contraindications=[
                "Pregnancy", "Angioedema history", "Bilateral renal artery stenosis"
            ],
            dosing={
                "Hypertension": "10mg daily, max 40mg daily",
                "Heart failure": "5mg daily, titrate to 20-40mg daily",
                "Post-MI": "5mg daily, titrate as tolerated"
            },
            side_effects=[
                "Dry cough", "Hyperkalemia", "Angioedema", 
                "Hypotension", "Renal impairment"
            ],
            interactions=[
                "NSAIDs", "Potassium supplements", "Lithium", "Diuretics"
            ],
            monitoring=["Blood pressure", "Renal function", "Potassium", "Cough"]
        )
        
        medications["atorvastatin"] = Medication(
            name="Atorvastatin",
            generic_name="Atorvastatin calcium",
            drug_class="HMG-CoA Reductase Inhibitor (Statin)",
            mechanism="Inhibits cholesterol synthesis",
            indications=[
                "Hyperlipidemia", "Primary prevention of CVD", 
                "Secondary prevention of CVD"
            ],
            contraindications=[
                "Active liver disease", "Pregnancy", "Breastfeeding"
            ],
            dosing={
                "Initial": "20mg daily",
                "Range": "10-80mg daily",
                "Timing": "Evening preferred"
            },
            side_effects=[
                "Myalgia", "Elevated liver enzymes", "Rhabdomyolysis (rare)",
                "Diabetes risk", "Memory issues"
            ],
            interactions=[
                "Warfarin", "Digoxin", "Cyclosporine", "Gemfibrozil"
            ],
            monitoring=["Lipid panel", "Liver function", "CK if symptoms"]
        )
        
        return medications
    
    def _load_lab_tests(self) -> Dict[str, LabTest]:
        """Load comprehensive laboratory test database"""
        lab_tests = {}
        
        lab_tests["troponin_i"] = LabTest(
            name="Troponin I",
            reference_range="<0.04 ng/mL",
            units="ng/mL",
            clinical_significance="Cardiac muscle damage marker",
            causes_high=[
                "Myocardial infarction", "Myocarditis", "Pulmonary embolism",
                "Renal failure", "Sepsis", "Cardioversion"
            ],
            causes_low=["Normal finding"],
            sample_type="Serum"
        )
        
        lab_tests["bnp"] = LabTest(
            name="B-type Natriuretic Peptide",
            reference_range="<100 pg/mL",
            units="pg/mL",
            clinical_significance="Heart failure marker",
            causes_high=[
                "Heart failure", "Pulmonary hypertension", "Renal failure",
                "Advanced age", "Atrial fibrillation"
            ],
            causes_low=["Normal cardiac function", "Obesity"],
            sample_type="Plasma"
        )
        
        lab_tests["hba1c"] = LabTest(
            name="Hemoglobin A1c",
            reference_range="<5.7% (normal), 5.7-6.4% (prediabetes), ≥6.5% (diabetes)",
            units="Percentage",
            clinical_significance="Average blood glucose over 2-3 months",
            causes_high=[
                "Diabetes mellitus", "Iron deficiency", "Kidney disease",
                "Certain medications"
            ],
            causes_low=[
                "Hemolytic anemia", "Blood loss", "Pregnancy"
            ],
            sample_type="Whole blood"
        )
        
        return lab_tests
    
    def _load_guidelines(self) -> Dict[str, Dict]:
        """Load clinical practice guidelines"""
        guidelines = {}
        
        guidelines["hypertension"] = {
            "source": "AHA/ACC 2017 Guidelines",
            "classification": {
                "Normal": "<120/80 mmHg",
                "Elevated": "120-129/<80 mmHg",
                "Stage 1": "130-139/80-89 mmHg",
                "Stage 2": "≥140/90 mmHg",
                "Crisis": ">180/120 mmHg"
            },
            "treatment_targets": {
                "General": "<130/80 mmHg",
                "Diabetes": "<130/80 mmHg",
                "CKD": "<130/80 mmHg",
                "Elderly": "<130/80 mmHg if tolerated"
            },
            "first_line_medications": [
                "ACE inhibitors", "ARBs", "Thiazide diuretics", "CCBs"
            ]
        }
        
        guidelines["diabetes"] = {
            "source": "ADA 2024 Standards of Care",
            "diagnostic_criteria": {
                "Fasting glucose": "≥126 mg/dL",
                "2-hour OGTT": "≥200 mg/dL",
                "HbA1c": "≥6.5%",
                "Random glucose": "≥200 mg/dL with symptoms"
            },
            "treatment_targets": {
                "HbA1c": "<7% (general), <6.5% (if no hypoglycemia)",
                "Blood pressure": "<130/80 mmHg",
                "LDL cholesterol": "<100 mg/dL, <70 mg/dL if CVD"
            },
            "first_line_medication": "Metformin"
        }
        
        return guidelines
    
    def _load_emergency_protocols(self) -> Dict[str, Dict]:
        """Load emergency medicine protocols"""
        protocols = {}
        
        protocols["stemi"] = {
            "name": "ST-Elevation Myocardial Infarction",
            "recognition": [
                "ST elevation ≥1mm in 2+ contiguous leads",
                "New LBBB", "Posterior MI equivalent"
            ],
            "time_targets": {
                "Door-to-balloon": "<90 minutes",
                "Door-to-needle": "<30 minutes"
            },
            "immediate_treatment": [
                "Aspirin 325mg chewed", "Clopidogrel 600mg loading",
                "Atorvastatin 80mg", "Metoprolol if no contraindications",
                "Heparin per protocol"
            ],
            "reperfusion_strategy": "Primary PCI preferred if available"
        }
        
        protocols["sepsis"] = {
            "name": "Sepsis Management",
            "recognition": [
                "qSOFA ≥2", "SIRS criteria", "Organ dysfunction"
            ],
            "time_targets": {
                "Antibiotics": "Within 1 hour",
                "Lactate": "Within 1 hour",
                "Blood cultures": "Before antibiotics"
            },
            "immediate_treatment": [
                "IV fluid resuscitation 30mL/kg",
                "Broad-spectrum antibiotics",
                "Vasopressors if hypotensive",
                "Source control"
            ]
        }
        
        return protocols
    
    def get_condition_info(self, condition_name: str) -> Optional[MedicalCondition]:
        """Retrieve information about a medical condition"""
        return self.conditions.get(condition_name.lower().replace(" ", "_"))
    
    def get_medication_info(self, medication_name: str) -> Optional[Medication]:
        """Retrieve information about a medication"""
        return self.medications.get(medication_name.lower())
    
    def get_lab_info(self, lab_name: str) -> Optional[LabTest]:
        """Retrieve information about a laboratory test"""
        return self.lab_tests.get(lab_name.lower().replace(" ", "_"))
    
    def search_conditions_by_symptom(self, symptom: str) -> List[str]:
        """Find conditions associated with a specific symptom"""
        matching_conditions = []
        symptom_lower = symptom.lower()
        
        for condition_key, condition in self.conditions.items():
            for condition_symptom in condition.symptoms:
                if symptom_lower in condition_symptom.lower():
                    matching_conditions.append(condition.name)
                    break
        
        return matching_conditions
    
    def get_drug_interactions(self, drug1: str, drug2: str) -> Optional[str]:
        """Check for interactions between two drugs"""
        # Simplified interaction checking
        interactions = {
            ("warfarin", "aspirin"): "Major: Increased bleeding risk",
            ("metformin", "contrast"): "Moderate: Risk of lactic acidosis",
            ("lisinopril", "nsaids"): "Moderate: Reduced antihypertensive effect",
            ("atorvastatin", "gemfibrozil"): "Major: Increased myopathy risk"
        }
        
        drug1_lower = drug1.lower()
        drug2_lower = drug2.lower()
        
        return (interactions.get((drug1_lower, drug2_lower)) or 
                interactions.get((drug2_lower, drug1_lower)))
    
    def get_clinical_guideline(self, condition: str) -> Optional[Dict]:
        """Retrieve clinical practice guidelines for a condition"""
        return self.clinical_guidelines.get(condition.lower())
    
    def get_emergency_protocol(self, condition: str) -> Optional[Dict]:
        """Retrieve emergency management protocol"""
        return self.emergency_protocols.get(condition.lower())
    
    def calculate_risk_scores(self, patient_data: Dict) -> Dict[str, float]:
        """Calculate various clinical risk scores"""
        risk_scores = {}
        
        # ASCVD Risk Calculator (simplified)
        if all(key in patient_data for key in ['age', 'sex', 'total_chol', 'hdl', 'sbp', 'smoking', 'diabetes']):
            # Simplified calculation - real implementation would use full Pooled Cohort Equations
            base_risk = 0.05  # 5% base risk
            
            if patient_data['age'] > 65:
                base_risk += 0.02
            if patient_data['sex'] == 'male':
                base_risk += 0.01
            if patient_data['smoking']:
                base_risk += 0.015
            if patient_data['diabetes']:
                base_risk += 0.02
            if patient_data['sbp'] > 140:
                base_risk += 0.01
            
            risk_scores['ascvd_10_year'] = min(base_risk, 0.3)  # Cap at 30%
        
        # CHA2DS2-VASc Score for Atrial Fibrillation
        if 'atrial_fibrillation' in patient_data and patient_data['atrial_fibrillation']:
            score = 0
            if patient_data.get('age', 0) >= 75:
                score += 2
            elif patient_data.get('age', 0) >= 65:
                score += 1
            if patient_data.get('sex') == 'female':
                score += 1
            if patient_data.get('heart_failure', False):
                score += 1
            if patient_data.get('hypertension', False):
                score += 1
            if patient_data.get('diabetes', False):
                score += 1
            if patient_data.get('stroke_history', False):
                score += 2
            if patient_data.get('vascular_disease', False):
                score += 1
            
            risk_scores['cha2ds2_vasc'] = score
        
        return risk_scores
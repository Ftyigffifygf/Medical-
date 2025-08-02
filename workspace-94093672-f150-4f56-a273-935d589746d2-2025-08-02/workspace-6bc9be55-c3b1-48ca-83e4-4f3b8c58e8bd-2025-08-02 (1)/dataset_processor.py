#!/usr/bin/env python3
"""
Medical Dataset Processor
Comprehensive data processing pipeline for medical datasets including:
- MIMIC-III/IV clinical records
- PubMed and PMC biomedical literature
- MedQuAD and MedRedQA question-answer datasets
- Medical dialogue datasets
- Medical imaging datasets
- BioBERT and medical NER processing
"""

import os
import json
import logging
import asyncio
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Generator
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
import zipfile
import gzip
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MedicalDataset:
    """Represents a medical dataset"""
    name: str
    source: str
    data_type: str
    size: int
    processed: bool = False
    file_path: Optional[str] = None

@dataclass
class ProcessedSample:
    """Represents a processed medical data sample"""
    id: str
    text: str
    labels: List[str]
    metadata: Dict[str, Any]
    dataset_source: str

class MedicalDatasetProcessor:
    """Comprehensive medical dataset processor"""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the dataset processor"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different dataset types
        self.subdirs = {
            'mimic': self.data_dir / 'mimic',
            'pubmed': self.data_dir / 'pubmed',
            'medquad': self.data_dir / 'medquad',
            'dialogues': self.data_dir / 'dialogues',
            'imaging': self.data_dir / 'imaging',
            'biobert': self.data_dir / 'biobert',
            'processed': self.data_dir / 'processed'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)
        
        self.datasets = {}
        logger.info("Medical Dataset Processor initialized")
    
    def download_pubmed_data(self, query: str, max_results: int = 1000) -> List[Dict]:
        """Download PubMed abstracts for a given query"""
        logger.info(f"Downloading PubMed data for query: {query}")
        
        # PubMed E-utilities API
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        # Search for PMIDs
        search_url = f"{base_url}esearch.fcgi"
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json'
        }
        
        try:
            search_response = requests.get(search_url, params=search_params)
            search_data = search_response.json()
            pmids = search_data['esearchresult']['idlist']
            
            logger.info(f"Found {len(pmids)} articles")
            
            # Fetch abstracts
            fetch_url = f"{base_url}efetch.fcgi"
            abstracts = []
            
            # Process in batches to avoid overwhelming the API
            batch_size = 100
            for i in range(0, len(pmids), batch_size):
                batch_pmids = pmids[i:i+batch_size]
                fetch_params = {
                    'db': 'pubmed',
                    'id': ','.join(batch_pmids),
                    'retmode': 'xml'
                }
                
                fetch_response = requests.get(fetch_url, params=fetch_params)
                
                # Parse XML response
                root = ET.fromstring(fetch_response.content)
                
                for article in root.findall('.//PubmedArticle'):
                    try:
                        pmid = article.find('.//PMID').text
                        title_elem = article.find('.//ArticleTitle')
                        abstract_elem = article.find('.//AbstractText')
                        
                        title = title_elem.text if title_elem is not None else ""
                        abstract = abstract_elem.text if abstract_elem is not None else ""
                        
                        if title or abstract:
                            abstracts.append({
                                'pmid': pmid,
                                'title': title,
                                'abstract': abstract,
                                'text': f"{title} {abstract}".strip()
                            })
                    except Exception as e:
                        logger.warning(f"Error processing article: {e}")
                        continue
            
            # Save to file
            output_file = self.subdirs['pubmed'] / f"pubmed_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json"
            with open(output_file, 'w') as f:
                json.dump(abstracts, f, indent=2)
            
            logger.info(f"Saved {len(abstracts)} abstracts to {output_file}")
            return abstracts
            
        except Exception as e:
            logger.error(f"Error downloading PubMed data: {e}")
            return []
    
    def download_medquad_dataset(self) -> bool:
        """Download MedQuAD dataset from Kaggle"""
        logger.info("Downloading MedQuAD dataset")
        
        # Note: This would require Kaggle API credentials
        # For now, we'll create a sample dataset structure
        sample_medquad = [
            {
                "id": "medquad_001",
                "question": "What are the symptoms of diabetes?",
                "answer": "Common symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, blurred vision, cuts/bruises that are slow to heal, weight loss, and tingling/numbness in hands/feet.",
                "category": "endocrine",
                "source": "NIH"
            },
            {
                "id": "medquad_002", 
                "question": "How is hypertension diagnosed?",
                "answer": "Hypertension is diagnosed when blood pressure readings consistently show systolic pressure ≥140 mmHg or diastolic pressure ≥90 mmHg on multiple occasions.",
                "category": "cardiovascular",
                "source": "AHA"
            },
            {
                "id": "medquad_003",
                "question": "What causes pneumonia?",
                "answer": "Pneumonia can be caused by bacteria (most commonly Streptococcus pneumoniae), viruses (influenza, RSV), fungi, or other microorganisms that infect the lungs.",
                "category": "respiratory", 
                "source": "CDC"
            }
        ]
        
        output_file = self.subdirs['medquad'] / "medquad_sample.json"
        with open(output_file, 'w') as f:
            json.dump(sample_medquad, f, indent=2)
        
        logger.info(f"Created sample MedQuAD dataset: {output_file}")
        return True
    
    def create_medical_dialogue_dataset(self) -> bool:
        """Create sample medical dialogue dataset"""
        logger.info("Creating medical dialogue dataset")
        
        sample_dialogues = [
            {
                "id": "dialogue_001",
                "conversation": [
                    {"speaker": "patient", "text": "I've been having chest pain for the past 2 days."},
                    {"speaker": "doctor", "text": "Can you describe the pain? Is it sharp, dull, or crushing?"},
                    {"speaker": "patient", "text": "It's a crushing pain that radiates to my left arm."},
                    {"speaker": "doctor", "text": "When did it start and what were you doing?"},
                    {"speaker": "patient", "text": "It started yesterday morning while I was walking upstairs."},
                    {"speaker": "doctor", "text": "We need to run some tests immediately. This could be a heart attack."}
                ],
                "diagnosis": "Myocardial Infarction",
                "urgency": "high"
            },
            {
                "id": "dialogue_002",
                "conversation": [
                    {"speaker": "patient", "text": "I have a persistent cough and fever."},
                    {"speaker": "doctor", "text": "How long have you had these symptoms?"},
                    {"speaker": "patient", "text": "About a week now. The cough is productive with yellow sputum."},
                    {"speaker": "doctor", "text": "Any shortness of breath or chest pain?"},
                    {"speaker": "patient", "text": "Yes, I get short of breath when walking."},
                    {"speaker": "doctor", "text": "Let's get a chest X-ray and some blood work to check for pneumonia."}
                ],
                "diagnosis": "Pneumonia",
                "urgency": "medium"
            }
        ]
        
        output_file = self.subdirs['dialogues'] / "medical_dialogues_sample.json"
        with open(output_file, 'w') as f:
            json.dump(sample_dialogues, f, indent=2)
        
        logger.info(f"Created medical dialogue dataset: {output_file}")
        return True
    
    def create_mimic_sample_dataset(self) -> bool:
        """Create sample MIMIC-style dataset"""
        logger.info("Creating MIMIC sample dataset")
        
        # Sample clinical notes in MIMIC format
        sample_mimic = [
            {
                "subject_id": "10001",
                "hadm_id": "20001", 
                "chartdate": "2024-01-15",
                "category": "Discharge summary",
                "text": """
CHIEF COMPLAINT: Chest pain and shortness of breath.

HISTORY OF PRESENT ILLNESS: 
65-year-old male with history of hypertension and diabetes presents with 2-day history of substernal chest pain radiating to left arm, associated with diaphoresis and nausea. Pain is 8/10, crushing in nature.

PAST MEDICAL HISTORY:
- Hypertension
- Type 2 Diabetes Mellitus
- Hyperlipidemia

MEDICATIONS:
- Metformin 1000mg BID
- Lisinopril 10mg daily
- Atorvastatin 40mg daily

PHYSICAL EXAMINATION:
Vital Signs: BP 160/95, HR 110, RR 22, O2 Sat 94% on RA
General: Diaphoretic, anxious appearing male
Cardiovascular: Tachycardic, regular rhythm, no murmurs
Pulmonary: Bilateral crackles at bases

LABORATORY DATA:
Troponin I: 15.2 ng/mL (elevated)
CK-MB: 45 ng/mL (elevated)
BNP: 850 pg/mL (elevated)

ASSESSMENT AND PLAN:
STEMI - ST elevation myocardial infarction
- Emergency cardiac catheterization
- Dual antiplatelet therapy
- Beta blocker, ACE inhibitor
- Statin therapy
                """,
                "diagnosis": "ST-elevation myocardial infarction"
            },
            {
                "subject_id": "10002",
                "hadm_id": "20002",
                "chartdate": "2024-01-16", 
                "category": "Physician notes",
                "text": """
CHIEF COMPLAINT: Fever, cough, and difficulty breathing.

HISTORY OF PRESENT ILLNESS:
45-year-old female presents with 5-day history of productive cough with purulent sputum, fever up to 102°F, and progressive dyspnea. No recent travel or sick contacts.

PAST MEDICAL HISTORY:
- Asthma
- GERD

PHYSICAL EXAMINATION:
Vital Signs: T 101.8°F, BP 125/80, HR 95, RR 24, O2 Sat 91% on RA
General: Ill-appearing, using accessory muscles
Pulmonary: Decreased breath sounds RLL, dullness to percussion

IMAGING:
Chest X-ray: Right lower lobe consolidation

LABORATORY:
WBC: 15,000 with left shift
Procalcitonin: 2.5 ng/mL

ASSESSMENT AND PLAN:
Community-acquired pneumonia
- Ceftriaxone and azithromycin
- Supplemental oxygen
- Respiratory therapy
                """,
                "diagnosis": "Community-acquired pneumonia"
            }
        ]
        
        output_file = self.subdirs['mimic'] / "mimic_sample.json"
        with open(output_file, 'w') as f:
            json.dump(sample_mimic, f, indent=2)
        
        logger.info(f"Created MIMIC sample dataset: {output_file}")
        return True
    
    def process_for_fine_tuning(self, dataset_type: str) -> List[Dict]:
        """Process datasets for OpenAI fine-tuning format"""
        logger.info(f"Processing {dataset_type} for fine-tuning")
        
        fine_tuning_data = []
        
        if dataset_type == "medquad":
            # Load MedQuAD data
            medquad_file = self.subdirs['medquad'] / "medquad_sample.json"
            if medquad_file.exists():
                with open(medquad_file, 'r') as f:
                    medquad_data = json.load(f)
                
                for item in medquad_data:
                    fine_tuning_example = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a medical AI assistant trained on comprehensive medical datasets. Provide accurate, evidence-based medical information."
                            },
                            {
                                "role": "user", 
                                "content": item["question"]
                            },
                            {
                                "role": "assistant",
                                "content": item["answer"]
                            }
                        ]
                    }
                    fine_tuning_data.append(fine_tuning_example)
        
        elif dataset_type == "dialogues":
            # Load dialogue data
            dialogue_file = self.subdirs['dialogues'] / "medical_dialogues_sample.json"
            if dialogue_file.exists():
                with open(dialogue_file, 'r') as f:
                    dialogue_data = json.load(f)
                
                for dialogue in dialogue_data:
                    # Convert dialogue to training format
                    conversation_text = ""
                    for turn in dialogue["conversation"]:
                        conversation_text += f"{turn['speaker']}: {turn['text']}\n"
                    
                    fine_tuning_example = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a medical AI assistant. Analyze medical conversations and provide appropriate diagnoses."
                            },
                            {
                                "role": "user",
                                "content": f"Analyze this medical conversation and provide a diagnosis:\n{conversation_text}"
                            },
                            {
                                "role": "assistant", 
                                "content": f"Based on the conversation, the likely diagnosis is: {dialogue['diagnosis']}"
                            }
                        ]
                    }
                    fine_tuning_data.append(fine_tuning_example)
        
        elif dataset_type == "mimic":
            # Load MIMIC data
            mimic_file = self.subdirs['mimic'] / "mimic_sample.json"
            if mimic_file.exists():
                with open(mimic_file, 'r') as f:
                    mimic_data = json.load(f)
                
                for record in mimic_data:
                    fine_tuning_example = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a medical AI assistant trained on clinical records. Analyze clinical notes and provide diagnoses."
                            },
                            {
                                "role": "user",
                                "content": f"Analyze this clinical note and provide the primary diagnosis:\n{record['text']}"
                            },
                            {
                                "role": "assistant",
                                "content": f"Primary diagnosis: {record['diagnosis']}"
                            }
                        ]
                    }
                    fine_tuning_data.append(fine_tuning_example)
        
        # Save processed data
        output_file = self.subdirs['processed'] / f"{dataset_type}_fine_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(output_file, 'w') as f:
            for example in fine_tuning_data:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Processed {len(fine_tuning_data)} examples for fine-tuning: {output_file}")
        return fine_tuning_data
    
    def create_comprehensive_dataset(self) -> str:
        """Create a comprehensive dataset combining all sources"""
        logger.info("Creating comprehensive medical dataset")
        
        # Download/create all datasets
        self.download_medquad_dataset()
        self.create_medical_dialogue_dataset()
        self.create_mimic_sample_dataset()
        
        # Process each dataset type
        all_fine_tuning_data = []
        
        for dataset_type in ["medquad", "dialogues", "mimic"]:
            data = self.process_for_fine_tuning(dataset_type)
            all_fine_tuning_data.extend(data)
        
        # Add PubMed data
        pubmed_queries = [
            "diabetes mellitus diagnosis treatment",
            "hypertension management guidelines", 
            "pneumonia symptoms diagnosis",
            "myocardial infarction treatment",
            "asthma management"
        ]
        
        for query in pubmed_queries:
            abstracts = self.download_pubmed_data(query, max_results=50)
            
            for abstract in abstracts:
                fine_tuning_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a medical AI assistant with access to biomedical literature. Provide evidence-based medical information."
                        },
                        {
                            "role": "user",
                            "content": f"Summarize the key medical information from this abstract: {abstract['title']}"
                        },
                        {
                            "role": "assistant",
                            "content": abstract['abstract']
                        }
                    ]
                }
                all_fine_tuning_data.append(fine_tuning_example)
        
        # Save comprehensive dataset
        comprehensive_file = self.subdirs['processed'] / f"comprehensive_medical_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(comprehensive_file, 'w') as f:
            for example in all_fine_tuning_data:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Created comprehensive dataset with {len(all_fine_tuning_data)} examples: {comprehensive_file}")
        return str(comprehensive_file)
    
    def validate_dataset(self, dataset_file: str) -> Dict[str, Any]:
        """Validate dataset for fine-tuning"""
        logger.info(f"Validating dataset: {dataset_file}")
        
        validation_results = {
            "total_examples": 0,
            "valid_examples": 0,
            "errors": [],
            "statistics": {}
        }
        
        try:
            with open(dataset_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    validation_results["total_examples"] += 1
                    
                    try:
                        example = json.loads(line)
                        
                        # Validate structure
                        if "messages" not in example:
                            validation_results["errors"].append(f"Line {line_num}: Missing 'messages' field")
                            continue
                        
                        messages = example["messages"]
                        if not isinstance(messages, list) or len(messages) < 2:
                            validation_results["errors"].append(f"Line {line_num}: Invalid messages format")
                            continue
                        
                        # Check message structure
                        valid_example = True
                        for msg in messages:
                            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                                validation_results["errors"].append(f"Line {line_num}: Invalid message structure")
                                valid_example = False
                                break
                        
                        if valid_example:
                            validation_results["valid_examples"] += 1
                    
                    except json.JSONDecodeError:
                        validation_results["errors"].append(f"Line {line_num}: Invalid JSON")
            
            # Calculate statistics
            validation_results["statistics"] = {
                "validity_rate": validation_results["valid_examples"] / validation_results["total_examples"] * 100,
                "error_count": len(validation_results["errors"])
            }
            
            logger.info(f"Validation complete: {validation_results['valid_examples']}/{validation_results['total_examples']} valid examples")
            
        except Exception as e:
            logger.error(f"Error validating dataset: {e}")
            validation_results["errors"].append(f"Validation error: {e}")
        
        return validation_results
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed datasets"""
        stats = {
            "datasets": {},
            "total_samples": 0,
            "processed_files": []
        }
        
        # Check each subdirectory
        for name, path in self.subdirs.items():
            if path.exists():
                files = list(path.glob("*.json")) + list(path.glob("*.jsonl"))
                stats["datasets"][name] = {
                    "file_count": len(files),
                    "files": [f.name for f in files]
                }
                
                # Count samples in JSON files
                sample_count = 0
                for file in files:
                    try:
                        if file.suffix == '.json':
                            with open(file, 'r') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    sample_count += len(data)
                                else:
                                    sample_count += 1
                        elif file.suffix == '.jsonl':
                            with open(file, 'r') as f:
                                sample_count += sum(1 for _ in f)
                    except Exception as e:
                        logger.warning(f"Error counting samples in {file}: {e}")
                
                stats["datasets"][name]["sample_count"] = sample_count
                stats["total_samples"] += sample_count
        
        return stats

def main():
    """Main function to demonstrate dataset processing"""
    processor = MedicalDatasetProcessor()
    
    # Create comprehensive dataset
    dataset_file = processor.create_comprehensive_dataset()
    
    # Validate the dataset
    validation_results = processor.validate_dataset(dataset_file)
    print(f"Dataset validation: {validation_results['valid_examples']}/{validation_results['total_examples']} valid examples")
    
    # Get statistics
    stats = processor.get_dataset_statistics()
    print(f"Total samples across all datasets: {stats['total_samples']}")
    
    return dataset_file

if __name__ == "__main__":
    main()
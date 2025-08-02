#!/usr/bin/env python3
"""
Fine-tuning Manager for Medical AI
Comprehensive fine-tuning pipeline for OpenAI models using medical datasets

Features:
- Dataset preparation and validation
- Fine-tuning job management
- Model evaluation and monitoring
- Performance tracking
- Model deployment and versioning
"""

import os
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from openai import OpenAI
import requests

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FineTuningJob:
    """Represents a fine-tuning job"""
    job_id: str
    model_name: str
    dataset_file: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    fine_tuned_model: Optional[str] = None
    training_file_id: Optional[str] = None
    validation_file_id: Optional[str] = None
    hyperparameters: Optional[Dict] = None
    metrics: Optional[Dict] = None

@dataclass
class ModelEvaluation:
    """Represents model evaluation results"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    clinical_accuracy: float
    safety_score: float
    evaluation_date: datetime
    test_cases: int

class FineTuningManager:
    """Manages fine-tuning operations for medical AI models"""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """Initialize the fine-tuning manager"""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        
        # Create directories
        for directory in [self.data_dir, self.models_dir]:
            directory.mkdir(exist_ok=True)
        
        self.jobs_file = self.models_dir / "fine_tuning_jobs.json"
        self.evaluations_file = self.models_dir / "model_evaluations.json"
        
        # Load existing jobs and evaluations
        self.jobs = self._load_jobs()
        self.evaluations = self._load_evaluations()
        
        logger.info("Fine-tuning Manager initialized")
    
    def _load_jobs(self) -> List[FineTuningJob]:
        """Load existing fine-tuning jobs"""
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file, 'r') as f:
                    jobs_data = json.load(f)
                
                jobs = []
                for job_data in jobs_data:
                    job_data['created_at'] = datetime.fromisoformat(job_data['created_at'])
                    if job_data.get('completed_at'):
                        job_data['completed_at'] = datetime.fromisoformat(job_data['completed_at'])
                    jobs.append(FineTuningJob(**job_data))
                
                return jobs
            except Exception as e:
                logger.error(f"Error loading jobs: {e}")
                return []
        return []
    
    def _save_jobs(self):
        """Save fine-tuning jobs to file"""
        try:
            jobs_data = []
            for job in self.jobs:
                job_dict = asdict(job)
                job_dict['created_at'] = job.created_at.isoformat()
                if job.completed_at:
                    job_dict['completed_at'] = job.completed_at.isoformat()
                jobs_data.append(job_dict)
            
            with open(self.jobs_file, 'w') as f:
                json.dump(jobs_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving jobs: {e}")
    
    def _load_evaluations(self) -> List[ModelEvaluation]:
        """Load model evaluations"""
        if self.evaluations_file.exists():
            try:
                with open(self.evaluations_file, 'r') as f:
                    eval_data = json.load(f)
                
                evaluations = []
                for eval_item in eval_data:
                    eval_item['evaluation_date'] = datetime.fromisoformat(eval_item['evaluation_date'])
                    evaluations.append(ModelEvaluation(**eval_item))
                
                return evaluations
            except Exception as e:
                logger.error(f"Error loading evaluations: {e}")
                return []
        return []
    
    def _save_evaluations(self):
        """Save model evaluations to file"""
        try:
            eval_data = []
            for evaluation in self.evaluations:
                eval_dict = asdict(evaluation)
                eval_dict['evaluation_date'] = evaluation.evaluation_date.isoformat()
                eval_data.append(eval_dict)
            
            with open(self.evaluations_file, 'w') as f:
                json.dump(eval_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving evaluations: {e}")
    
    def prepare_dataset(self, dataset_file: str, validation_split: float = 0.1) -> Tuple[str, str]:
        """Prepare dataset for fine-tuning with train/validation split"""
        logger.info(f"Preparing dataset: {dataset_file}")
        
        try:
            # Load dataset
            examples = []
            with open(dataset_file, 'r') as f:
                for line in f:
                    examples.append(json.loads(line))
            
            # Shuffle examples
            np.random.shuffle(examples)
            
            # Split into train/validation
            split_idx = int(len(examples) * (1 - validation_split))
            train_examples = examples[:split_idx]
            val_examples = examples[split_idx:]
            
            # Save training set
            train_file = self.data_dir / f"train_{Path(dataset_file).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            with open(train_file, 'w') as f:
                for example in train_examples:
                    f.write(json.dumps(example) + '\n')
            
            # Save validation set
            val_file = self.data_dir / f"val_{Path(dataset_file).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            with open(val_file, 'w') as f:
                for example in val_examples:
                    f.write(json.dumps(example) + '\n')
            
            logger.info(f"Dataset prepared: {len(train_examples)} training, {len(val_examples)} validation examples")
            return str(train_file), str(val_file)
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise
    
    def upload_dataset(self, dataset_file: str) -> str:
        """Upload dataset to OpenAI"""
        logger.info(f"Uploading dataset: {dataset_file}")
        
        try:
            with open(dataset_file, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose='fine-tune'
                )
            
            file_id = response.id
            logger.info(f"Dataset uploaded successfully: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Error uploading dataset: {e}")
            raise
    
    def start_fine_tuning(
        self,
        training_file: str,
        model_name: str,
        validation_file: Optional[str] = None,
        hyperparameters: Optional[Dict] = None
    ) -> str:
        """Start a fine-tuning job"""
        logger.info(f"Starting fine-tuning job for model: {model_name}")
        
        try:
            # Upload training file
            training_file_id = self.upload_dataset(training_file)
            
            # Upload validation file if provided
            validation_file_id = None
            if validation_file:
                validation_file_id = self.upload_dataset(validation_file)
            
            # Set default hyperparameters
            if hyperparameters is None:
                hyperparameters = {
                    "n_epochs": 3,
                    "batch_size": 1,
                    "learning_rate_multiplier": 0.1
                }
            
            # Create fine-tuning job
            job_params = {
                "training_file": training_file_id,
                "model": "gpt-3.5-turbo",
                "suffix": model_name,
                "hyperparameters": hyperparameters
            }
            
            if validation_file_id:
                job_params["validation_file"] = validation_file_id
            
            response = self.client.fine_tuning.jobs.create(**job_params)
            
            job_id = response.id
            
            # Create job record
            job = FineTuningJob(
                job_id=job_id,
                model_name=model_name,
                dataset_file=training_file,
                status="running",
                created_at=datetime.now(),
                training_file_id=training_file_id,
                validation_file_id=validation_file_id,
                hyperparameters=hyperparameters
            )
            
            self.jobs.append(job)
            self._save_jobs()
            
            logger.info(f"Fine-tuning job started: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error starting fine-tuning: {e}")
            raise
    
    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """Check the status of a fine-tuning job"""
        try:
            response = self.client.fine_tuning.jobs.retrieve(job_id)
            
            status_info = {
                "id": response.id,
                "status": response.status,
                "created_at": response.created_at,
                "finished_at": response.finished_at,
                "fine_tuned_model": response.fine_tuned_model,
                "training_file": response.training_file,
                "validation_file": response.validation_file,
                "hyperparameters": response.hyperparameters,
                "result_files": response.result_files,
                "trained_tokens": response.trained_tokens
            }
            
            # Update local job record
            for job in self.jobs:
                if job.job_id == job_id:
                    job.status = response.status
                    if response.finished_at:
                        job.completed_at = datetime.fromtimestamp(response.finished_at)
                    if response.fine_tuned_model:
                        job.fine_tuned_model = response.fine_tuned_model
                    break
            
            self._save_jobs()
            return status_info
            
        except Exception as e:
            logger.error(f"Error checking job status: {e}")
            raise
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all fine-tuning jobs"""
        job_list = []
        for job in self.jobs:
            job_dict = asdict(job)
            job_dict['created_at'] = job.created_at.isoformat()
            if job.completed_at:
                job_dict['completed_at'] = job.completed_at.isoformat()
            job_list.append(job_dict)
        
        return job_list
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a fine-tuning job"""
        try:
            self.client.fine_tuning.jobs.cancel(job_id)
            
            # Update local job record
            for job in self.jobs:
                if job.job_id == job_id:
                    job.status = "cancelled"
                    break
            
            self._save_jobs()
            logger.info(f"Fine-tuning job cancelled: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling job: {e}")
            return False
    
    def evaluate_model(self, model_name: str, test_cases: List[Dict]) -> ModelEvaluation:
        """Evaluate a fine-tuned model"""
        logger.info(f"Evaluating model: {model_name}")
        
        try:
            correct_predictions = 0
            total_predictions = len(test_cases)
            
            # Clinical accuracy metrics
            clinical_correct = 0
            safety_violations = 0
            
            for i, test_case in enumerate(test_cases):
                try:
                    # Get model prediction
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=test_case["messages"][:-1],  # Exclude the expected answer
                        temperature=0.1,
                        max_tokens=500
                    )
                    
                    prediction = response.choices[0].message.content
                    expected = test_case["messages"][-1]["content"]
                    
                    # Simple accuracy check (can be improved with more sophisticated metrics)
                    if self._compare_medical_responses(prediction, expected):
                        correct_predictions += 1
                    
                    # Clinical accuracy check
                    if self._check_clinical_accuracy(prediction, expected):
                        clinical_correct += 1
                    
                    # Safety check
                    if self._check_safety_violations(prediction):
                        safety_violations += 1
                    
                except Exception as e:
                    logger.warning(f"Error evaluating test case {i}: {e}")
                    continue
            
            # Calculate metrics
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            clinical_accuracy = clinical_correct / total_predictions if total_predictions > 0 else 0
            safety_score = 1 - (safety_violations / total_predictions) if total_predictions > 0 else 1
            
            # For now, using simplified precision/recall/f1
            precision = accuracy  # Simplified
            recall = accuracy     # Simplified
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            evaluation = ModelEvaluation(
                model_name=model_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                clinical_accuracy=clinical_accuracy,
                safety_score=safety_score,
                evaluation_date=datetime.now(),
                test_cases=total_predictions
            )
            
            self.evaluations.append(evaluation)
            self._save_evaluations()
            
            logger.info(f"Model evaluation complete: Accuracy={accuracy:.3f}, Clinical Accuracy={clinical_accuracy:.3f}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def _compare_medical_responses(self, prediction: str, expected: str) -> bool:
        """Compare medical responses for accuracy"""
        # Simplified comparison - can be enhanced with medical NLP
        prediction_lower = prediction.lower()
        expected_lower = expected.lower()
        
        # Check for key medical terms overlap
        medical_terms = ["diagnosis", "treatment", "symptoms", "medication", "condition"]
        
        prediction_terms = set()
        expected_terms = set()
        
        for term in medical_terms:
            if term in prediction_lower:
                prediction_terms.add(term)
            if term in expected_lower:
                expected_terms.add(term)
        
        # Simple overlap check
        if len(expected_terms) == 0:
            return True
        
        overlap = len(prediction_terms.intersection(expected_terms))
        return overlap / len(expected_terms) >= 0.5
    
    def _check_clinical_accuracy(self, prediction: str, expected: str) -> bool:
        """Check clinical accuracy of prediction"""
        # Simplified clinical accuracy check
        # In practice, this would use medical knowledge bases and clinical guidelines
        
        prediction_lower = prediction.lower()
        expected_lower = expected.lower()
        
        # Check for dangerous contradictions
        dangerous_terms = ["contraindicated", "dangerous", "harmful", "toxic"]
        
        for term in dangerous_terms:
            if term in expected_lower and term not in prediction_lower:
                return False
            if term in prediction_lower and term not in expected_lower:
                return False
        
        return True
    
    def _check_safety_violations(self, prediction: str) -> bool:
        """Check for safety violations in prediction"""
        prediction_lower = prediction.lower()
        
        # Check for unsafe recommendations
        unsafe_patterns = [
            "ignore symptoms",
            "don't see a doctor",
            "self-medicate",
            "increase dosage without",
            "stop medication immediately"
        ]
        
        for pattern in unsafe_patterns:
            if pattern in prediction_lower:
                return True
        
        return False
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of all evaluated models"""
        if not self.evaluations:
            return {"message": "No model evaluations available"}
        
        # Calculate summary statistics
        accuracies = [eval.accuracy for eval in self.evaluations]
        clinical_accuracies = [eval.clinical_accuracy for eval in self.evaluations]
        safety_scores = [eval.safety_score for eval in self.evaluations]
        
        summary = {
            "total_models_evaluated": len(self.evaluations),
            "average_accuracy": np.mean(accuracies),
            "average_clinical_accuracy": np.mean(clinical_accuracies),
            "average_safety_score": np.mean(safety_scores),
            "best_model": max(self.evaluations, key=lambda x: x.clinical_accuracy).model_name,
            "latest_evaluation": max(self.evaluations, key=lambda x: x.evaluation_date).evaluation_date.isoformat(),
            "model_rankings": sorted(
                [(eval.model_name, eval.clinical_accuracy, eval.safety_score) for eval in self.evaluations],
                key=lambda x: (x[1], x[2]),
                reverse=True
            )
        }
        
        return summary
    
    def create_test_dataset(self) -> List[Dict]:
        """Create a test dataset for model evaluation"""
        test_cases = [
            {
                "messages": [
                    {"role": "system", "content": "You are a medical AI assistant."},
                    {"role": "user", "content": "A 65-year-old male presents with chest pain radiating to the left arm, diaphoresis, and nausea. What is the most likely diagnosis?"},
                    {"role": "assistant", "content": "Based on the symptoms described - chest pain radiating to the left arm, diaphoresis (sweating), and nausea in a 65-year-old male - the most likely diagnosis is myocardial infarction (heart attack). This presentation is classic for acute coronary syndrome. Immediate medical attention is required, including ECG, cardiac enzymes, and emergency cardiac catheterization if indicated."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a medical AI assistant."},
                    {"role": "user", "content": "What are the typical symptoms of pneumonia?"},
                    {"role": "assistant", "content": "Typical symptoms of pneumonia include: fever and chills, cough (often productive with purulent sputum), shortness of breath or difficulty breathing, chest pain that worsens with breathing or coughing, fatigue and weakness, nausea and vomiting, and confusion (especially in elderly patients). Symptoms may vary depending on the causative organism and patient's age and health status."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a medical AI assistant."},
                    {"role": "user", "content": "How is diabetes mellitus diagnosed?"},
                    {"role": "assistant", "content": "Diabetes mellitus is diagnosed using one of the following criteria: 1) Fasting plasma glucose ≥126 mg/dL (7.0 mmol/L) on two separate occasions, 2) Random plasma glucose ≥200 mg/dL (11.1 mmol/L) with symptoms of diabetes, 3) 2-hour plasma glucose ≥200 mg/dL during oral glucose tolerance test, or 4) Hemoglobin A1C ≥6.5% (48 mmol/mol). The diagnosis should be confirmed with repeat testing unless there are unequivocal symptoms of hyperglycemia."}
                ]
            }
        ]
        
        return test_cases
    
    def monitor_all_jobs(self) -> Dict[str, Any]:
        """Monitor all active fine-tuning jobs"""
        active_jobs = [job for job in self.jobs if job.status in ["running", "pending"]]
        
        monitoring_results = {
            "total_jobs": len(self.jobs),
            "active_jobs": len(active_jobs),
            "completed_jobs": len([job for job in self.jobs if job.status == "succeeded"]),
            "failed_jobs": len([job for job in self.jobs if job.status == "failed"]),
            "job_updates": []
        }
        
        # Check status of active jobs
        for job in active_jobs:
            try:
                status_info = self.check_job_status(job.job_id)
                monitoring_results["job_updates"].append({
                    "job_id": job.job_id,
                    "model_name": job.model_name,
                    "status": status_info["status"],
                    "fine_tuned_model": status_info.get("fine_tuned_model")
                })
            except Exception as e:
                logger.error(f"Error monitoring job {job.job_id}: {e}")
        
        return monitoring_results

def main():
    """Main function to demonstrate fine-tuning manager"""
    manager = FineTuningManager()
    
    # Example usage
    print("Fine-tuning Manager initialized")
    print(f"Total jobs: {len(manager.jobs)}")
    print(f"Total evaluations: {len(manager.evaluations)}")
    
    # Monitor jobs
    monitoring_results = manager.monitor_all_jobs()
    print(f"Active jobs: {monitoring_results['active_jobs']}")
    
    # Get performance summary
    performance_summary = manager.get_model_performance_summary()
    print("Performance Summary:", performance_summary)

if __name__ == "__main__":
    main()
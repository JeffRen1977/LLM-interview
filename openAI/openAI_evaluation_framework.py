#!/usr/bin/env python3
"""
OpenAI Evaluation Framework Implementation

Complete implementation of evaluation framework for generative models,
including automated metrics, human evaluation simulation, and online evaluation.

Author: Extracted from openAI_evaluation_framework.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import json
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Try to import evaluation libraries
try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    print("Warning: evaluate library not available. Install with: pip install evaluate")
    EVALUATE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    print("Warning: bert_score library not available. Install with: pip install bert-score")
    BERT_SCORE_AVAILABLE = False


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    metric_name: str
    score: float
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None


class EvaluationMetric(ABC):
    """Abstract base class for evaluation metrics"""
    
    @abstractmethod
    def compute(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        pass


class PerplexityMetric(EvaluationMetric):
    """Perplexity metric for text generation models"""
    
    def __init__(self, model=None):
        self.model = model
    
    def compute(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        """
        Compute perplexity for generated text
        Note: This is a simplified implementation. In practice, you'd use a trained model.
        """
        # Simplified perplexity calculation
        # In reality, you'd use a language model to compute actual perplexity
        avg_length = np.mean([len(text.split()) for text in predictions])
        # Simulate perplexity based on text length and complexity
        simulated_ppl = 50 + np.random.normal(0, 10) + avg_length * 0.5
        
        return EvaluationResult(
            metric_name="Perplexity",
            score=simulated_ppl,
            metadata={"avg_length": avg_length}
        )


class BLEUMetric(EvaluationMetric):
    """BLEU metric for text generation"""
    
    def __init__(self, n_gram=4):
        self.n_gram = n_gram
    
    def compute(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        """Compute BLEU score"""
        if not EVALUATE_AVAILABLE:
            # Fallback implementation
            return self._compute_bleu_fallback(predictions, references)
        
        try:
            bleu = evaluate.load("bleu")
            results = bleu.compute(predictions=predictions, references=references)
            return EvaluationResult(
                metric_name="BLEU",
                score=results["bleu"],
                metadata={"n_gram": self.n_gram}
            )
        except Exception as e:
            print(f"Error computing BLEU: {e}")
            return self._compute_bleu_fallback(predictions, references)
    
    def _compute_bleu_fallback(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        """Fallback BLEU computation"""
        # Simplified BLEU calculation
        scores = []
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # Simple n-gram overlap
            overlap = len(set(pred_words) & set(ref_words))
            precision = overlap / len(pred_words) if pred_words else 0
            scores.append(precision)
        
        avg_score = np.mean(scores)
        return EvaluationResult(
            metric_name="BLEU",
            score=avg_score,
            metadata={"method": "fallback"}
        )


class ROUGEMetric(EvaluationMetric):
    """ROUGE metric for text generation"""
    
    def __init__(self, rouge_type="rougeL"):
        self.rouge_type = rouge_type
    
    def compute(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        """Compute ROUGE score"""
        if not EVALUATE_AVAILABLE:
            return self._compute_rouge_fallback(predictions, references)
        
        try:
            rouge = evaluate.load("rouge")
            results = rouge.compute(predictions=predictions, references=references)
            return EvaluationResult(
                metric_name="ROUGE",
                score=results["rougeL"],
                metadata={"type": self.rouge_type}
            )
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            return self._compute_rouge_fallback(predictions, references)
    
    def _compute_rouge_fallback(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        """Fallback ROUGE computation"""
        scores = []
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # Simple recall calculation
            overlap = len(set(pred_words) & set(ref_words))
            recall = overlap / len(ref_words) if ref_words else 0
            scores.append(recall)
        
        avg_score = np.mean(scores)
        return EvaluationResult(
            metric_name="ROUGE",
            score=avg_score,
            metadata={"method": "fallback"}
        )


class BERTScoreMetric(EvaluationMetric):
    """BERTScore metric for semantic similarity"""
    
    def compute(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        """Compute BERTScore"""
        if not BERT_SCORE_AVAILABLE:
            return self._compute_bertscore_fallback(predictions, references)
        
        try:
            P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
            return EvaluationResult(
                metric_name="BERTScore",
                score=F1.mean().item(),
                metadata={
                    "precision": P.mean().item(),
                    "recall": R.mean().item(),
                    "f1": F1.mean().item()
                }
            )
        except Exception as e:
            print(f"Error computing BERTScore: {e}")
            return self._compute_bertscore_fallback(predictions, references)
    
    def _compute_bertscore_fallback(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        """Fallback BERTScore computation using simple word overlap"""
        scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            if not pred_words or not ref_words:
                scores.append(0.0)
                continue
            
            precision = len(pred_words & ref_words) / len(pred_words)
            recall = len(pred_words & ref_words) / len(ref_words)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            scores.append(f1)
        
        avg_score = np.mean(scores)
        return EvaluationResult(
            metric_name="BERTScore",
            score=avg_score,
            metadata={"method": "fallback"}
        )


class HumanEvaluationSimulator:
    """Simulate human evaluation for demonstration purposes"""
    
    def __init__(self):
        self.evaluation_dimensions = [
            "fluency", "coherence", "factuality", 
            "helpfulness", "harmlessness", "creativity"
        ]
    
    def ab_test(self, model_a_outputs: List[str], model_b_outputs: List[str], 
                num_evaluators: int = 10) -> Dict:
        """Simulate A/B testing between two models"""
        results = {"model_a_wins": 0, "model_b_wins": 0, "ties": 0}
        
        for _ in range(num_evaluators):
            for output_a, output_b in zip(model_a_outputs, model_b_outputs):
                # Simulate evaluator preference based on text quality
                score_a = self._evaluate_text_quality(output_a)
                score_b = self._evaluate_text_quality(output_b)
                
                if score_a > score_b + 0.1:  # Add some threshold for ties
                    results["model_a_wins"] += 1
                elif score_b > score_a + 0.1:
                    results["model_b_wins"] += 1
                else:
                    results["ties"] += 1
        
        total = sum(results.values())
        results["model_a_win_rate"] = results["model_a_wins"] / total
        results["model_b_win_rate"] = results["model_b_wins"] / total
        results["tie_rate"] = results["ties"] / total
        
        return results
    
    def likert_rating(self, outputs: List[str], dimension: str = "overall") -> Dict:
        """Simulate Likert scale ratings"""
        ratings = []
        for output in outputs:
            # Simulate rating based on text quality
            quality_score = self._evaluate_text_quality(output)
            # Convert to 1-5 scale
            rating = max(1, min(5, int(quality_score * 5)))
            ratings.append(rating)
        
        return {
            "dimension": dimension,
            "ratings": ratings,
            "average": np.mean(ratings),
            "std": np.std(ratings),
            "distribution": {i: ratings.count(i) for i in range(1, 6)}
        }
    
    def _evaluate_text_quality(self, text: str) -> float:
        """Simulate text quality evaluation"""
        # Simple heuristics for text quality
        length_score = min(1.0, len(text.split()) / 50)  # Prefer moderate length
        complexity_score = len(set(text.lower().split())) / len(text.split()) if text.split() else 0
        readability_score = 1.0 - min(1.0, len([w for w in text.split() if len(w) > 10]) / len(text.split()))
        
        # Combine scores with some randomness
        base_score = (length_score + complexity_score + readability_score) / 3
        noise = np.random.normal(0, 0.1)
        return max(0, min(1, base_score + noise))


class OnlineEvaluationSimulator:
    """Simulate online evaluation metrics"""
    
    def __init__(self):
        self.user_behavior_patterns = {
            "engagement": 0.7,  # 70% of users engage
            "satisfaction": 0.8,  # 80% satisfaction rate
            "retention": 0.6,  # 60% retention rate
        }
    
    def collect_implicit_signals(self, outputs: List[str], num_users: int = 100) -> Dict:
        """Simulate collection of implicit user signals"""
        signals = {
            "thumbs_up": 0,
            "thumbs_down": 0,
            "copied_responses": 0,
            "session_duration": [],
            "follow_up_questions": 0,
        }
        
        for output in outputs:
            for _ in range(num_users):
                # Simulate user behavior based on output quality
                quality = self._estimate_output_quality(output)
                
                # Thumbs up/down
                if np.random.random() < quality:
                    signals["thumbs_up"] += 1
                else:
                    signals["thumbs_down"] += 1
                
                # Copy behavior
                if quality > 0.7 and np.random.random() < 0.3:
                    signals["copied_responses"] += 1
                
                # Session duration (in seconds)
                base_duration = 30 + quality * 60
                duration = max(10, base_duration + np.random.normal(0, 15))
                signals["session_duration"].append(duration)
                
                # Follow-up questions
                if quality < 0.5 and np.random.random() < 0.4:
                    signals["follow_up_questions"] += 1
        
        # Calculate averages
        signals["avg_session_duration"] = np.mean(signals["session_duration"])
        signals["satisfaction_rate"] = signals["thumbs_up"] / (signals["thumbs_up"] + signals["thumbs_down"])
        
        return signals
    
    def ab_test_online(self, model_a_outputs: List[str], model_b_outputs: List[str], 
                      traffic_split: float = 0.1) -> Dict:
        """Simulate online A/B testing"""
        results = {
            "model_a": {"users": 0, "satisfaction": 0, "retention": 0},
            "model_b": {"users": 0, "satisfaction": 0, "retention": 0}
        }
        
        total_users = 1000
        model_a_users = int(total_users * traffic_split)
        model_b_users = total_users - model_a_users
        
        # Model A results
        for _ in range(model_a_users):
            for output in model_a_outputs:
                quality = self._estimate_output_quality(output)
                results["model_a"]["users"] += 1
                if np.random.random() < quality:
                    results["model_a"]["satisfaction"] += 1
                if np.random.random() < self.user_behavior_patterns["retention"]:
                    results["model_a"]["retention"] += 1
        
        # Model B results
        for _ in range(model_b_users):
            for output in model_b_outputs:
                quality = self._estimate_output_quality(output)
                results["model_b"]["users"] += 1
                if np.random.random() < quality:
                    results["model_b"]["satisfaction"] += 1
                if np.random.random() < self.user_behavior_patterns["retention"]:
                    results["model_b"]["retention"] += 1
        
        # Calculate rates
        for model in ["model_a", "model_b"]:
            if results[model]["users"] > 0:
                results[model]["satisfaction_rate"] = results[model]["satisfaction"] / results[model]["users"]
                results[model]["retention_rate"] = results[model]["retention"] / results[model]["users"]
        
        return results
    
    def _estimate_output_quality(self, text: str) -> float:
        """Estimate output quality for simulation"""
        # Simple quality estimation
        length_score = min(1.0, len(text.split()) / 30)
        diversity_score = len(set(text.lower().split())) / len(text.split()) if text.split() else 0
        return (length_score + diversity_score) / 2


class EvaluationFramework:
    """Complete evaluation framework for generative models"""
    
    def __init__(self):
        self.automated_metrics = {
            "perplexity": PerplexityMetric(),
            "bleu": BLEUMetric(),
            "rouge": ROUGEMetric(),
            "bertscore": BERTScoreMetric(),
        }
        self.human_evaluator = HumanEvaluationSimulator()
        self.online_evaluator = OnlineEvaluationSimulator()
    
    def evaluate_automated_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """Run all automated metrics"""
        results = {}
        
        print("🤖 Running automated metrics...")
        for name, metric in self.automated_metrics.items():
            try:
                result = metric.compute(predictions, references)
                results[name] = result
                print(f"  ✅ {name}: {result.score:.4f}")
            except Exception as e:
                print(f"  ❌ {name}: Error - {e}")
                results[name] = EvaluationResult(name, 0.0, metadata={"error": str(e)})
        
        return results
    
    def evaluate_human_metrics(self, predictions: List[str], references: List[str] = None) -> Dict:
        """Run human evaluation simulation"""
        print("👥 Running human evaluation simulation...")
        
        # Simulate A/B test (comparing with references if available)
        if references:
            ab_results = self.human_evaluator.ab_test(predictions, references)
            print(f"  📊 A/B Test - Model A: {ab_results['model_a_win_rate']:.2%}, "
                  f"Model B: {ab_results['model_b_win_rate']:.2%}")
        else:
            ab_results = None
        
        # Likert ratings
        likert_results = {}
        for dimension in self.human_evaluator.evaluation_dimensions:
            result = self.human_evaluator.likert_rating(predictions, dimension)
            likert_results[dimension] = result
            print(f"  📈 {dimension.capitalize()}: {result['average']:.2f} ± {result['std']:.2f}")
        
        return {
            "ab_test": ab_results,
            "likert_ratings": likert_results
        }
    
    def evaluate_online_metrics(self, predictions: List[str]) -> Dict:
        """Run online evaluation simulation"""
        print("🌐 Running online evaluation simulation...")
        
        # Implicit signals
        implicit_signals = self.online_evaluator.collect_implicit_signals(predictions)
        print(f"  👍 Satisfaction rate: {implicit_signals['satisfaction_rate']:.2%}")
        print(f"  📋 Copied responses: {implicit_signals['copied_responses']}")
        print(f"  ⏱️  Avg session duration: {implicit_signals['avg_session_duration']:.1f}s")
        
        return {"implicit_signals": implicit_signals}
    
    def comprehensive_evaluation(self, predictions: List[str], references: List[str] = None) -> Dict:
        """Run comprehensive evaluation across all dimensions"""
        print("🚀 Starting comprehensive evaluation...")
        print("=" * 60)
        
        results = {}
        
        # Automated metrics
        results["automated"] = self.evaluate_automated_metrics(predictions, references or predictions)
        
        # Human evaluation
        results["human"] = self.evaluate_human_metrics(predictions, references)
        
        # Online evaluation
        results["online"] = self.evaluate_online_metrics(predictions)
        
        print("=" * 60)
        print("✅ Comprehensive evaluation completed!")
        
        return results
    
    def generate_report(self, results: Dict, save_path: str = None) -> str:
        """Generate evaluation report"""
        report = []
        report.append("# Evaluation Report")
        report.append("=" * 50)
        
        # Automated metrics summary
        report.append("## Automated Metrics")
        for name, result in results["automated"].items():
            report.append(f"- **{name.upper()}**: {result.score:.4f}")
        
        # Human evaluation summary
        report.append("\n## Human Evaluation")
        if results["human"]["ab_test"]:
            ab = results["human"]["ab_test"]
            report.append(f"- **A/B Test Win Rate**: {ab['model_a_win_rate']:.2%}")
        
        report.append("- **Likert Ratings**:")
        for dim, rating in results["human"]["likert_ratings"].items():
            report.append(f"  - {dim.capitalize()}: {rating['average']:.2f}")
        
        # Online evaluation summary
        report.append("\n## Online Evaluation")
        signals = results["online"]["implicit_signals"]
        report.append(f"- **Satisfaction Rate**: {signals['satisfaction_rate']:.2%}")
        report.append(f"- **Avg Session Duration**: {signals['avg_session_duration']:.1f}s")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Report saved to: {save_path}")
        
        return report_text


def demo_evaluation_framework():
    """Demonstrate the evaluation framework"""
    
    print("🧠 OpenAI Evaluation Framework Demo")
    print("=" * 60)
    
    # Sample data
    predictions = [
        "The cat is sitting on the mat.",
        "Machine learning is a subset of artificial intelligence.",
        "The weather today is sunny and warm.",
        "Python is a popular programming language for data science.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    references = [
        "There is a cat on the mat.",
        "ML is part of AI technology.",
        "Today's weather is pleasant and sunny.",
        "Python programming is widely used in data analysis.",
        "A brown fox quickly jumps over a sleeping dog."
    ]
    
    # Initialize framework
    framework = EvaluationFramework()
    
    # Run comprehensive evaluation
    results = framework.comprehensive_evaluation(predictions, references)
    
    # Generate report
    report = framework.generate_report(results, "evaluation_report.md")
    
    print("\n📊 Evaluation Summary:")
    print(report)
    
    return results


def demo_bert_score_example():
    """Demonstrate BERTScore calculation as shown in the original markdown"""
    
    print("\n🔍 BERTScore Example (from markdown)")
    print("=" * 60)
    
    # Example from the markdown
    predictions = ["The cat is on the mat."]
    references = [
        "There is a cat on the mat.", 
        "A cat is lying on the rug."
    ]
    
    # Use our BERTScore implementation
    bertscore_metric = BERTScoreMetric()
    result = bertscore_metric.compute(predictions, references)
    
    print(f"Predictions: {predictions}")
    print(f"References: {references}")
    print(f"\nBERTScore Results:")
    print(f"Precision: {result.metadata.get('precision', result.score):.4f}")
    print(f"Recall: {result.metadata.get('recall', result.score):.4f}")
    print(f"F1 Score: {result.metadata.get('f1', result.score):.4f}")
    
    return result


def main():
    """Main function to run all demonstrations"""
    
    print("🎯 OpenAI Evaluation Framework Implementation")
    print("Extracted and expanded from openAI_evaluation_framework.md")
    print("=" * 60)
    
    # Check available libraries
    if EVALUATE_AVAILABLE:
        print("✅ evaluate library available")
    else:
        print("⚠️  evaluate library not available - using fallback implementations")
    
    if BERT_SCORE_AVAILABLE:
        print("✅ bert_score library available")
    else:
        print("⚠️  bert_score library not available - using fallback implementations")
    
    print()
    
    # Run demonstrations
    try:
        # Demo 1: BERTScore example from markdown
        demo_bert_score_example()
        
        # Demo 2: Comprehensive evaluation framework
        demo_evaluation_framework()
        
        print("\n" + "=" * 60)
        print("🎉 All demonstrations completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

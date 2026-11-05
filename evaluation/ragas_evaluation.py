
-"""
RAGAS Evaluation Framework for Multilingual RAG System
Evaluates retrieval quality and answer generation using RAGAS metrics

Metrics Evaluated:
- Faithfulness: Is the answer grounded in retrieved context?
- Answer Relevance: Does the answer address the query?
- Context Precision: Are retrieved documents relevant?
- Context Recall: Are all relevant documents retrieved?
"""

import os
import sys
from pathlib import Path
import json
import time
from typing import List, Dict, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance
)

# Import our RAG system
from src.core.qa import retrieve_candidates, answer_query
from src.core.config import CFG


class RAGASEvaluator:
    """
    Evaluates RAG system using RAGAS framework
    """
    
    def __init__(self):
        print("🚀 Initializing RAGAS Evaluator...")
        
        # Test that QA functions are accessible
        print("✅ QA functions loaded successfully\n")
    
    def load_test_queries(self, queries_file: str = None) -> List[Dict[str, Any]]:
        """
        Load test queries from markdown file or use predefined set
        """
        if queries_file and Path(queries_file).exists():
            print(f"Loading queries from {queries_file}")
            # Parse markdown file (simplified for now)
            queries = self._parse_queries_from_markdown(queries_file)
        else:
            # Use predefined test queries
            queries = self._get_predefined_queries()
        
        print(f"Loaded {len(queries)} test queries")
        return queries
    
    def _get_predefined_queries(self) -> List[Dict[str, Any]]:
        """
        Predefined test queries covering different categories
        """
        return [
            # Direct ID lookups
            {
                "query": "Was ist DTAD-ID 20046891?",
                "category": "direct_lookup",
                "expected_docs": ["20046891"]
            },
            {
                "query": "Show me tender 20047108",
                "category": "direct_lookup",
                "expected_docs": ["20047108"]
            },
            
            # Temporal queries
            {
                "query": "Ausschreibungen aus 2024",
                "category": "temporal",
                "expected_count": 10
            },
            {
                "query": "Recent tenders",
                "category": "temporal",
                "expected_count": 5
            },
            
            # Semantic/Conceptual
            {
                "query": "IT-Projekte und Software-Entwicklung",
                "category": "semantic",
                "keywords": ["IT", "software", "development"]
            },
            {
                "query": "Bauprojekte in Deutschland",
                "category": "semantic",
                "keywords": ["construction", "building", "Germany"]
            },
            {
                "query": "Lieferung von medizinischen Geräten",
                "category": "semantic",
                "keywords": ["medical", "equipment", "delivery"]
            },
            
            # Multilingual
            {
                "query": "IT projects in Berlin",
                "category": "multilingual",
                "keywords": ["IT", "Berlin"]
            },
            {
                "query": "Software development tenders",
                "category": "multilingual",
                "keywords": ["software", "development"]
            },
            
            # Complex questions
            {
                "query": "Was sind die Anforderungen für IT-Projekte?",
                "category": "qa",
                "expected_answer_contains": ["Anforderungen", "IT"]
            },
            {
                "query": "Welche Fristen gelten für Ausschreibungen?",
                "category": "qa",
                "expected_answer_contains": ["Fristen", "deadline"]
            }
        ]
    
    def _parse_queries_from_markdown(self, filepath: str) -> List[Dict]:
        """Parse queries from SAMPLE_QUERIES.md"""
        queries = []
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Simple parsing - extract numbered queries
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and '`' in line:
                # Extract query between backticks
                start = line.find('`') + 1
                end = line.find('`', start)
                if start > 0 and end > start:
                    query = line[start:end]
                    queries.append({"query": query, "category": "general"})
        
        return queries
    
    def prepare_dataset(self, queries: List[Dict[str, Any]]) -> Dataset:
        """
        Prepare dataset in RAGAS format
        
        RAGAS expects:
        - question: The user query
        - answer: Generated answer from LLM
        - contexts: List of retrieved document texts
        - ground_truth: Expected answer (optional, needed for context_recall)
        """
        print("\n📊 Generating responses for all queries...")
        
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []  # We'll use the query as ground truth for now
        }
        
        for i, q_dict in enumerate(queries, 1):
            query = q_dict["query"]
            print(f"\n[{i}/{len(queries)}] Processing: {query}")
            
            try:
                # Retrieve documents using QA function
                candidates = retrieve_candidates(query, cfg=CFG, limit=5)
                
                # Extract context texts
                contexts = [hit.text for hit in candidates]
                
                # Generate answer using QA function
                answer = answer_query(query, cfg=CFG)
                
                # Add to dataset
                data["question"].append(query)
                data["answer"].append(answer)
                data["contexts"].append(contexts)
                
                # Use query as ground truth (in real scenario, use human-labeled answers)
                data["ground_truth"].append(f"Answer related to: {query}")
                
                print(f"   ✓ Retrieved {len(contexts)} documents")
                print(f"   ✓ Answer: {answer[:100]}...")
                
            except Exception as e:
                print(f"   ✗ Error: {e}")
                # Add empty data to maintain dataset consistency
                data["question"].append(query)
                data["answer"].append("Error generating answer")
                data["contexts"].append([])
                data["ground_truth"].append("")
        
        # Create HuggingFace Dataset
        dataset = Dataset.from_dict(data)
        print(f"\n✅ Dataset prepared: {len(dataset)} samples")
        
        return dataset
    
    def evaluate_system(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Run RAGAS evaluation
        """
        print("\n🔬 Running RAGAS Evaluation...")
        print("=" * 60)
        
        # Select metrics
        metrics = [
            Faithfulness(),           # Answer grounded in context
            AnswerRelevancy(),        # Answer addresses question
            ContextPrecision(),       # Retrieved docs are relevant
            # ContextRecall(),        # All relevant docs retrieved (needs ground_truth)
            ContextRelevance()        # Context is relevant to question
        ]
        
        print("Metrics:")
        for metric in metrics:
            print(f"  - {metric.name}")
        
        print("\nEvaluating...")
        start_time = time.time()
        
        try:
            # Run evaluation
            results = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm="ollama/qwen2.5:1.5b",  # Use Ollama for evaluation
            )
            
            eval_time = time.time() - start_time
            
            print(f"\n✅ Evaluation complete in {eval_time:.2f}s")
            print("=" * 60)
            
            return {
                "results": results,
                "eval_time": eval_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"\n❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def print_results(self, eval_results: Dict[str, Any]):
        """Print formatted evaluation results"""
        if not eval_results:
            print("No results to display")
            return
        
        results = eval_results["results"]
        
        print("\n" + "=" * 60)
        print("📈 RAGAS EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"\nTimestamp: {eval_results['timestamp']}")
        print(f"Evaluation Time: {eval_results['eval_time']:.2f}s")
        
        print("\n📊 Metric Scores:")
        print("-" * 60)
        
        # Print individual metric scores
        for key, value in results.items():
            if isinstance(value, (int, float)):
                print(f"  {key:.<40} {value:.4f}")
        
        print("-" * 60)
        
        # Calculate average
        metric_scores = [v for v in results.values() if isinstance(v, (int, float))]
        if metric_scores:
            avg_score = sum(metric_scores) / len(metric_scores)
            print(f"  {'Average Score':.<40} {avg_score:.4f}")
        
        print("=" * 60)
        
        # Interpretation
        print("\n💡 Interpretation:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                if value >= 0.8:
                    status = "✅ Excellent"
                elif value >= 0.6:
                    status = "⚠️  Good"
                elif value >= 0.4:
                    status = "⚠️  Fair"
                else:
                    status = "❌ Needs Improvement"
                
                print(f"  {key}: {status} ({value:.2%})")
    
    def save_results(self, eval_results: Dict[str, Any], output_dir: str = "evaluation/results"):
        """Save results to JSON file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"ragas_eval_{timestamp}.json"
        
        # Convert results to serializable format
        results_dict = {
            "timestamp": eval_results["timestamp"],
            "eval_time": eval_results["eval_time"],
            "scores": {}
        }
        
        for key, value in eval_results["results"].items():
            if isinstance(value, (int, float)):
                results_dict["scores"][key] = value
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Main evaluation pipeline"""
    print("=" * 60)
    print("🎯 RAGAS EVALUATION - Multilingual RAG System")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = RAGASEvaluator()
    
    # Load test queries
    queries_file = Path("evaluation/datasets/SAMPLE_QUERIES.md")
    if queries_file.exists():
        test_queries = evaluator.load_test_queries(str(queries_file))
    else:
        print("⚠️  SAMPLE_QUERIES.md not found, using predefined queries")
        test_queries = evaluator.load_test_queries()
    
    # Limit to first 10 queries for quick evaluation
    test_queries = test_queries[:10]
    print(f"\n📋 Using {len(test_queries)} queries for evaluation")
    
    # Prepare dataset
    dataset = evaluator.prepare_dataset(test_queries)
    
    # Run evaluation
    eval_results = evaluator.evaluate_system(dataset)
    
    if eval_results:
        # Print results
        evaluator.print_results(eval_results)
        
        # Save results
        evaluator.save_results(eval_results)
        
        print("\n✅ Evaluation complete!")
        print("\n💡 Next steps:")
        print("  1. Review results to identify weaknesses")
        print("  2. Rebuild vector database if needed")
        print("  3. Test with full 87 queries")
        print("  4. Use for interview presentation")
    else:
        print("\n❌ Evaluation failed. Check logs above.")


if __name__ == "__main__":
    main()

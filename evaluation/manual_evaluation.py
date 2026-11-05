"""
Manual RAG Evaluation Framework
Comprehensive testing without external API dependencies

Tests:
1. Retrieval Quality (keyword matching, semantic search)
2. Answer Generation (completeness, relevance)
3. End-to-End Performance (latency, throughput)
4. Robustness (error handling, edge cases)
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.qa import retrieve_candidates, answer_query
from src.core.search import search_hybrid, search_dense
from src.core.config import CFG


class ManualRAGEvaluator:
    """
    Manual RAG evaluation without external API dependencies
    """
    
    def __init__(self):
        print("🚀 Initializing Manual RAG Evaluator...")
        print("=" * 70)
        
        # Test categories
        self.test_queries = {
            "exact_keyword": [
                ("IT-Projekte", ["IT", "Projekte", "software"]),
                ("Bauprojekte", ["Bau", "construction", "building"]),
                ("Lieferung", ["Liefer", "delivery", "supply"]),
            ],
            "semantic": [
                ("Software development projects", ["software", "development", "IT"]),
                ("Construction tenders in Germany", ["construction", "building", "Germany"]),
                ("Medical equipment supply", ["medical", "equipment", "supply"]),
            ],
            "id_lookup": [
                ("DTAD-ID 20046891", ["20046891"]),
                ("Tender 20047108", ["20047108"]),
                ("Vergabe 20046893", ["20046893"]),
            ],
            "complex_qa": [
                ("Was sind die Anforderungen für IT-Projekte?", ["Anforderung", "IT"]),
                ("Welche Fristen gelten für Ausschreibungen?", ["Frist", "deadline"]),
                ("Wie läuft ein Vergabeverfahren ab?", ["Vergabe", "procedure"]),
            ]
        }
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "categories": {},
            "summary": {}
        }
    
    def evaluate_retrieval_quality(self) -> Dict[str, Any]:
        """
        Test 1: Retrieval Quality
        Measures how well the system retrieves relevant documents
        """
        print("\n" + "=" * 70)
        print("TEST 1: RETRIEVAL QUALITY")
        print("=" * 70)
        
        category_results = {}
        
        for category, queries in self.test_queries.items():
            print(f"\n{category.upper()}:")
            print("-" * 70)
            
            category_scores = []
            
            for query, expected_terms in queries:
                print(f"\nQuery: {query}")
                
                start_time = time.time()
                try:
                    # Retrieve documents
                    results = retrieve_candidates(query, cfg=CFG, limit=5)
                    retrieval_time = time.time() - start_time
                    
                    # Check if expected terms appear in results
                    found_terms = set()
                    for hit in results:
                        text = hit.text.lower()
                        for term in expected_terms:
                            if term.lower() in text:
                                found_terms.add(term)
                    
                    relevance_score = len(found_terms) / len(expected_terms) if expected_terms else 0
                    
                    print(f"  ✓ Retrieved: {len(results)} documents in {retrieval_time:.3f}s")
                    print(f"  ✓ Top score: {results[0].score:.4f}")
                    print(f"  ✓ Relevance: {relevance_score:.1%} ({len(found_terms)}/{len(expected_terms)} terms found)")
                    
                    category_scores.append({
                        "query": query,
                        "num_results": len(results),
                        "retrieval_time": retrieval_time,
                        "top_score": float(results[0].score) if results else 0.0,
                        "relevance_score": relevance_score,
                        "found_terms": list(found_terms)
                    })
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    category_scores.append({
                        "query": query,
                        "error": str(e)
                    })
            
            # Calculate category average
            valid_scores = [s for s in category_scores if "error" not in s]
            if valid_scores:
                avg_relevance = sum(s["relevance_score"] for s in valid_scores) / len(valid_scores)
                avg_time = sum(s["retrieval_time"] for s in valid_scores) / len(valid_scores)
                avg_top_score = sum(s["top_score"] for s in valid_scores) / len(valid_scores)
                
                print(f"\n  Category Average:")
                print(f"    Relevance: {avg_relevance:.1%}")
                print(f"    Time: {avg_time:.3f}s")
                print(f"    Top Score: {avg_top_score:.4f}")
                
                category_results[category] = {
                    "queries": category_scores,
                    "avg_relevance": avg_relevance,
                    "avg_time": avg_time,
                    "avg_top_score": avg_top_score
                }
            else:
                category_results[category] = {
                    "queries": category_scores,
                    "avg_relevance": 0.0,
                    "error": "All queries failed"
                }
        
        return category_results
    
    def evaluate_answer_generation(self) -> Dict[str, Any]:
        """
        Test 2: Answer Generation Quality
        Measures answer completeness and relevance
        """
        print("\n" + "=" * 70)
        print("TEST 2: ANSWER GENERATION QUALITY")
        print("=" * 70)
        
        qa_queries = self.test_queries["complex_qa"]
        results = []
        
        for query, expected_terms in qa_queries:
            print(f"\nQuery: {query}")
            print("-" * 70)
            
            start_time = time.time()
            try:
                # Generate answer
                answer = answer_query(query, cfg=CFG)
                generation_time = time.time() - start_time
                
                # Check answer quality
                answer_lower = answer.lower()
                found_terms = sum(1 for term in expected_terms if term.lower() in answer_lower)
                term_coverage = found_terms / len(expected_terms) if expected_terms else 0
                
                # Check answer length (non-empty, not too short)
                is_adequate_length = len(answer.split()) >= 10
                
                # Check if answer admits uncertainty properly
                uncertainty_phrases = ["nicht", "keine", "unknown", "I don't know"]
                admits_uncertainty = any(phrase in answer_lower for phrase in uncertainty_phrases)
                
                print(f"  Answer: {answer[:200]}...")
                print(f"  ✓ Generated in: {generation_time:.2f}s")
                print(f"  ✓ Length: {len(answer.split())} words")
                print(f"  ✓ Term coverage: {term_coverage:.1%}")
                print(f"  ✓ Adequate length: {is_adequate_length}")
                
                results.append({
                    "query": query,
                    "answer": answer,
                    "generation_time": generation_time,
                    "word_count": len(answer.split()),
                    "term_coverage": term_coverage,
                    "adequate_length": is_adequate_length,
                    "admits_uncertainty": admits_uncertainty
                })
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({
                    "query": query,
                    "error": str(e)
                })
        
        # Calculate averages
        valid_results = [r for r in results if "error" not in r]
        if valid_results:
            avg_time = sum(r["generation_time"] for r in valid_results) / len(valid_results)
            avg_coverage = sum(r["term_coverage"] for r in valid_results) / len(valid_results)
            pct_adequate = sum(1 for r in valid_results if r["adequate_length"]) / len(valid_results)
            
            print(f"\n  Overall Answer Quality:")
            print(f"    Avg generation time: {avg_time:.2f}s")
            print(f"    Avg term coverage: {avg_coverage:.1%}")
            print(f"    Adequate length: {pct_adequate:.1%}")
            
            return {
                "queries": results,
                "avg_generation_time": avg_time,
                "avg_term_coverage": avg_coverage,
                "pct_adequate_length": pct_adequate
            }
        else:
            return {"queries": results, "error": "All queries failed"}
    
    def evaluate_performance(self) -> Dict[str, Any]:
        """
        Test 3: Performance & Throughput
        Measures system responsiveness under load
        """
        print("\n" + "=" * 70)
        print("TEST 3: PERFORMANCE & THROUGHPUT")
        print("=" * 70)
        
        test_query = "IT-Projekte"
        num_iterations = 10
        
        print(f"\nRunning {num_iterations} iterations with query: '{test_query}'")
        print("-" * 70)
        
        retrieval_times = []
        generation_times = []
        
        for i in range(num_iterations):
            # Test retrieval
            start = time.time()
            results = retrieve_candidates(test_query, cfg=CFG, limit=5)
            retrieval_time = time.time() - start
            retrieval_times.append(retrieval_time)
            
            # Test answer generation (every 3rd iteration to avoid overwhelming)
            if i % 3 == 0:
                start = time.time()
                answer = answer_query(test_query, cfg=CFG)
                generation_time = time.time() - start
                generation_times.append(generation_time)
            
            print(f"  Iteration {i+1}: Retrieval={retrieval_time:.3f}s", end="")
            if i % 3 == 0 and generation_times:
                print(f", Generation={generation_times[-1]:.2f}s")
            else:
                print()
        
        # Calculate statistics
        avg_retrieval = sum(retrieval_times) / len(retrieval_times)
        min_retrieval = min(retrieval_times)
        max_retrieval = max(retrieval_times)
        
        avg_generation = sum(generation_times) / len(generation_times) if generation_times else 0
        
        print(f"\n  Performance Statistics:")
        print(f"    Retrieval - Avg: {avg_retrieval:.3f}s, Min: {min_retrieval:.3f}s, Max: {max_retrieval:.3f}s")
        print(f"    Generation - Avg: {avg_generation:.2f}s ({len(generation_times)} samples)")
        print(f"    Throughput - Retrieval: {1/avg_retrieval:.1f} queries/sec")
        
        return {
            "retrieval": {
                "avg": avg_retrieval,
                "min": min_retrieval,
                "max": max_retrieval,
                "throughput_qps": 1/avg_retrieval
            },
            "generation": {
                "avg": avg_generation,
                "samples": len(generation_times)
            }
        }
    
    def evaluate_robustness(self) -> Dict[str, Any]:
        """
        Test 4: Robustness & Error Handling
        Tests edge cases and error scenarios
        """
        print("\n" + "=" * 70)
        print("TEST 4: ROBUSTNESS & ERROR HANDLING")
        print("=" * 70)
        
        edge_cases = [
            ("", "Empty query"),
            ("x", "Single character"),
            ("a b c d e f", "Very short tokens"),
            ("q" * 1000, "Very long single token"),
            ("🎯 🚀 ✅", "Only emojis"),
            ("12345678", "Only numbers (possible ID)"),
            ("SELECT * FROM users", "SQL injection attempt"),
        ]
        
        results = []
        
        for query, description in edge_cases:
            print(f"\nTest: {description}")
            print(f"  Query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
            
            try:
                start = time.time()
                hits = retrieve_candidates(query, cfg=CFG, limit=3)
                elapsed = time.time() - start
                
                print(f"  ✓ Handled gracefully")
                print(f"  ✓ Results: {len(hits)}, Time: {elapsed:.3f}s")
                
                results.append({
                    "description": description,
                    "query": query[:100],
                    "status": "success",
                    "num_results": len(hits),
                    "time": elapsed
                })
                
            except Exception as e:
                print(f"  ✗ Error: {type(e).__name__}: {str(e)[:100]}")
                results.append({
                    "description": description,
                    "query": query[:100],
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error_msg": str(e)[:200]
                })
        
        success_rate = sum(1 for r in results if r["status"] == "success") / len(results)
        
        print(f"\n  Robustness Summary:")
        print(f"    Success rate: {success_rate:.1%}")
        print(f"    Total tests: {len(results)}")
        
        return {
            "tests": results,
            "success_rate": success_rate,
            "total_tests": len(results)
        }
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """
        Run all evaluation tests
        """
        print("\n" + "=" * 70)
        print("🎯 MANUAL RAG EVALUATION - COMPREHENSIVE TESTING")
        print("=" * 70)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"System: Multilingual RAG (Hybrid Search + Ollama LLM)")
        
        # Run all tests
        self.results["categories"]["retrieval"] = self.evaluate_retrieval_quality()
        self.results["categories"]["generation"] = self.evaluate_answer_generation()
        self.results["categories"]["performance"] = self.evaluate_performance()
        self.results["categories"]["robustness"] = self.evaluate_robustness()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def generate_summary(self):
        """Generate evaluation summary"""
        print("\n" + "=" * 70)
        print("📊 EVALUATION SUMMARY")
        print("=" * 70)
        
        retrieval = self.results["categories"]["retrieval"]
        generation = self.results["categories"]["generation"]
        performance = self.results["categories"]["performance"]
        robustness = self.results["categories"]["robustness"]
        
        # Calculate overall scores
        retrieval_avg = []
        for cat in retrieval.values():
            if "avg_relevance" in cat:
                retrieval_avg.append(cat["avg_relevance"])
        
        overall_retrieval = sum(retrieval_avg) / len(retrieval_avg) if retrieval_avg else 0
        overall_generation = generation.get("avg_term_coverage", 0)
        overall_robustness = robustness.get("success_rate", 0)
        
        print(f"\n📈 Overall Scores:")
        print(f"  Retrieval Quality:    {overall_retrieval:.1%}")
        print(f"  Answer Generation:    {overall_generation:.1%}")
        print(f"  Robustness:           {overall_robustness:.1%}")
        print(f"\n⚡ Performance:")
        print(f"  Avg Retrieval Time:   {performance['retrieval']['avg']:.3f}s")
        print(f"  Avg Generation Time:  {performance['generation']['avg']:.2f}s")
        print(f"  Retrieval Throughput: {performance['retrieval']['throughput_qps']:.1f} q/s")
        
        # Store summary
        self.results["summary"] = {
            "overall_retrieval_quality": overall_retrieval,
            "overall_generation_quality": overall_generation,
            "overall_robustness": overall_robustness,
            "avg_retrieval_time": performance["retrieval"]["avg"],
            "avg_generation_time": performance["generation"]["avg"],
            "throughput_qps": performance["retrieval"]["throughput_qps"]
        }
        
        # Grade system
        def grade(score):
            if score >= 0.9: return "A (Excellent)"
            elif score >= 0.8: return "B (Very Good)"
            elif score >= 0.7: return "C (Good)"
            elif score >= 0.6: return "D (Satisfactory)"
            else: return "F (Needs Improvement)"
        
        print(f"\n🎯 System Grades:")
        print(f"  Retrieval:  {grade(overall_retrieval)}")
        print(f"  Generation: {grade(overall_generation)}")
        print(f"  Robustness: {grade(overall_robustness)}")
    
    def save_results(self):
        """Save evaluation results to file"""
        output_dir = Path("evaluation/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"manual_eval_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Run manual RAG evaluation"""
    evaluator = ManualRAGEvaluator()
    results = evaluator.run_full_evaluation()
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)
    print("\nYour RAG system has been comprehensively tested!")
    print("Check evaluation/results/ for detailed JSON reports.")


if __name__ == "__main__":
    main()

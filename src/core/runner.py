import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class EvaluationRunner:
    """
    Core runner that orchestrates evaluations across different domains and languages.
    """
    
    def __init__(self):
        self.evaluators = {}
        
    def register_evaluator(self, name: str, evaluator: Any):
        self.evaluators[name] = evaluator
        
    def evaluate(self, domain: str, generated_text: str, expected_tests: str) -> Dict[str, Any]:
        """
        Runs the domain-specific evaluator on the generated text.
        """
        if domain not in self.evaluators:
            logger.error(f"No evaluator found for domain: {domain}")
            return {"status": "error", "message": f"Unknown domain: {domain}"}
            
        evaluator = self.evaluators[domain]
        logger.info(f"Running evaluation for {domain}...")
        
        result = evaluator.run(generated_text, expected_tests)
        return result

    def batch_evaluate(self, domain: str, generations: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Runs evaluation on a batch of generations.
        generations: List of {"generated": "...", "tests": "..."}
        """
        logger.info(f"Starting batch evaluation for {len(generations)} samples in {domain}.")
        results = []
        passed = 0
        
        for item in generations:
            res = self.evaluate(domain, item["generated"], item["tests"])
            results.append(res)
            if res.get("status") == "pass":
                passed += 1
                
        accuracy = (passed / len(generations)) * 100 if generations else 0
        logger.info(f"Batch evaluation complete. Pass rate: {accuracy:.2f}% ({passed}/{len(generations)})")
        
        return {
            "total": len(generations),
            "passed": passed,
            "accuracy": accuracy,
            "details": results
        }

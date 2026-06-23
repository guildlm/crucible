import os
import tempfile
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GoEvaluator:
    """
    Evaluates generated Go code by compiling and running unit tests against it.
    """
    
    def run(self, generated_code: str, expected_tests: str) -> Dict[str, Any]:
        """
        Runs `go test` on the generated code.
        """
        # Create a temporary sandbox
        with tempfile.TemporaryDirectory() as temp_dir:
            main_file = os.path.join(temp_dir, "main.go")
            test_file = os.path.join(temp_dir, "main_test.go")
            
            # Write files
            with open(main_file, "w") as f:
                f.write(generated_code)
                
            with open(test_file, "w") as f:
                f.write(expected_tests)
                
            # Initialize go module
            try:
                subprocess.run(
                    ["go", "mod", "init", "sandbox"],
                    cwd=temp_dir,
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError as e:
                return {"status": "error", "message": "Failed to init go module", "stderr": e.stderr.decode()}
                
            # Run tests
            try:
                result = subprocess.run(
                    ["go", "test", "-v"],
                    cwd=temp_dir,
                    capture_output=True,
                    timeout=10 # Prevent infinite loops
                )
                
                stdout = result.stdout.decode('utf-8', errors='replace')
                stderr = result.stderr.decode('utf-8', errors='replace')
                
                if result.returncode == 0:
                    return {
                        "status": "pass",
                        "output": stdout
                    }
                else:
                    return {
                        "status": "fail",
                        "output": stdout,
                        "stderr": stderr
                    }
            except subprocess.TimeoutExpired:
                return {
                    "status": "fail",
                    "message": "Execution timed out (10s)",
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": str(e)
                }

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    evaluator = GoEvaluator()
    
    good_code = "package sandbox\n\nfunc Add(a, b int) int { return a + b }"
    bad_code = "package sandbox\n\nfunc Add(a, b int) int { return a - b }"
    tests = '''package sandbox
import "testing"
func TestAdd(t *testing.T) {
    if Add(2, 3) != 5 { t.Fatal("Failed") }
}'''

    logger.info("Testing Good Code:")
    res_good = evaluator.run(good_code, tests)
    print(res_good)
    
    logger.info("Testing Bad Code:")
    res_bad = evaluator.run(bad_code, tests)
    print(res_bad)

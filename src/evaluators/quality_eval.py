import tempfile
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GoQualityEvaluator:
    """
    Evaluates generated Go code for code quality (linting, vetting, static analysis).
    """
    def run(self, generated_code: str) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/main.go"
            with open(file_path, "w") as f:
                f.write(generated_code)
                
            try:
                # Basic vet check
                res = subprocess.run(["go", "vet", file_path], capture_output=True, text=True)
                if res.returncode == 0:
                    return {"status": "pass", "message": "Code is clean."}
                else:
                    return {"status": "fail", "message": "Linting failed", "details": res.stderr}
            except Exception as e:
                return {"status": "error", "message": str(e)}

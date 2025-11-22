"""
Safe wrapper for running diagnostics with proper encoding
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Force UTF-8 encoding for stdout/stderr on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    
    # Set console code page to UTF-8
    os.system('chcp 65001 > nul')

# Now import and run the diagnostics
from diagnostics.run_full_diagnostic import FullDiagnostic

if __name__ == "__main__":
    diagnostic = FullDiagnostic()
    diagnostic.run_all_tests()
    
    print("\n" + "="*80)
    print("FULL DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nResults saved to:")
    print("  - diagnostics/results/full_diagnostic.json")
    print("  - diagnostics/results/latency_profile.json")
    print("  - diagnostics/results/retrieval_quality.json")
    print("  - diagnostics/results/response_quality.json")
    print("  - diagnostics/results/RECOMMENDATIONS.txt")
    print("\n" + "="*80)


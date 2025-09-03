#!/usr/bin/env python3
"""
Comprehensive Test Script for All OpenAI Interview Questions

This script tests all the OpenAI interview question implementations in the openAI directory.
It runs each script and captures the output, providing a summary of results.

Author: Generated for testing OpenAI interview implementations
"""

import os
import sys
import subprocess
import time
import traceback
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class OpenAITestRunner:
    """Test runner for all OpenAI interview question scripts"""
    
    def __init__(self, openai_dir: str = "openAI"):
        self.openai_dir = openai_dir
        self.results = {}
        self.start_time = datetime.now()
        
    def get_python_files(self) -> List[str]:
        """Get all Python files in the openAI directory"""
        python_files = []
        if os.path.exists(self.openai_dir):
            for file in os.listdir(self.openai_dir):
                if file.endswith('.py') and file.startswith('openAI_'):
                    python_files.append(file)
        return sorted(python_files)
    
    def run_script(self, script_name: str) -> Tuple[bool, str, str]:
        """Run a single Python script and capture output"""
        script_path = os.path.join(self.openai_dir, script_name)
        
        if not os.path.exists(script_path):
            return False, f"Script not found: {script_path}", ""
        
        try:
            print(f"\n{'='*80}")
            print(f"Running: {script_name}")
            print(f"{'='*80}")
            
            # Run the script with timeout
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=os.getcwd()
            )
            
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr
            
            if success:
                print(f"✅ {script_name} - SUCCESS")
                if stdout:
                    print("Output:")
                    print(stdout)
            else:
                print(f"❌ {script_name} - FAILED (exit code: {result.returncode})")
                if stderr:
                    print("Error:")
                    print(stderr)
                if stdout:
                    print("Output:")
                    print(stdout)
            
            return success, stdout, stderr
            
        except subprocess.TimeoutExpired:
            error_msg = f"Script timed out after 5 minutes"
            print(f"⏰ {script_name} - TIMEOUT")
            return False, "", error_msg
            
        except Exception as e:
            error_msg = f"Exception occurred: {str(e)}\n{traceback.format_exc()}"
            print(f"💥 {script_name} - EXCEPTION")
            print(f"Error: {error_msg}")
            return False, "", error_msg
    
    def run_all_tests(self) -> Dict[str, Dict]:
        """Run all OpenAI scripts and collect results"""
        python_files = self.get_python_files()
        
        if not python_files:
            print(f"No Python files found in {self.openai_dir} directory")
            return {}
        
        print(f"Found {len(python_files)} Python files to test:")
        for file in python_files:
            print(f"  - {file}")
        
        for script_name in python_files:
            start_time = time.time()
            success, stdout, stderr = self.run_script(script_name)
            end_time = time.time()
            
            self.results[script_name] = {
                'success': success,
                'stdout': stdout,
                'stderr': stderr,
                'execution_time': end_time - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Small delay between tests
            time.sleep(1)
        
        return self.results
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report"""
        total_scripts = len(self.results)
        successful_scripts = sum(1 for result in self.results.values() if result['success'])
        failed_scripts = total_scripts - successful_scripts
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        report = f"""
{'='*80}
OPENAI INTERVIEW SCRIPTS TEST SUMMARY
{'='*80}

Test Execution Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
Total Execution Duration: {total_time:.2f} seconds
Total Scripts Tested: {total_scripts}
Successful Scripts: {successful_scripts}
Failed Scripts: {failed_scripts}
Success Rate: {(successful_scripts/total_scripts*100):.1f}%

{'='*80}
DETAILED RESULTS
{'='*80}
"""
        
        for script_name, result in self.results.items():
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            exec_time = f"{result['execution_time']:.2f}s"
            
            report += f"\n{script_name}:\n"
            report += f"  Status: {status}\n"
            report += f"  Execution Time: {exec_time}\n"
            report += f"  Timestamp: {result['timestamp']}\n"
            
            if not result['success'] and result['stderr']:
                report += f"  Error: {result['stderr'][:200]}...\n" if len(result['stderr']) > 200 else f"  Error: {result['stderr']}\n"
        
        # Add recommendations
        report += f"""
{'='*80}
RECOMMENDATIONS
{'='*80}
"""
        
        if failed_scripts > 0:
            report += f"""
⚠️  {failed_scripts} script(s) failed. Please check the error messages above.
Common issues might include:
- Missing dependencies
- Import errors
- Runtime errors
- Memory issues
- Timeout issues

"""
        
        if successful_scripts == total_scripts:
            report += """
🎉 All scripts executed successfully! 
The OpenAI interview implementations are working correctly.
"""
        
        report += f"""
📊 Performance Summary:
- Average execution time per script: {sum(r['execution_time'] for r in self.results.values())/total_scripts:.2f}s
- Fastest script: {min(self.results.items(), key=lambda x: x[1]['execution_time'])[0]} ({min(r['execution_time'] for r in self.results.values()):.2f}s)
- Slowest script: {max(self.results.items(), key=lambda x: x[1]['execution_time'])[0]} ({max(r['execution_time'] for r in self.results.values()):.2f}s)
"""
        
        return report
    
    def save_report(self, filename: str = None):
        """Save the test report to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"openai_test_report_{timestamp}.txt"
        
        report = self.generate_summary_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Test report saved to: {filename}")
        return filename

def main():
    """Main function to run all tests"""
    print("🚀 Starting OpenAI Interview Scripts Test Suite")
    print("=" * 80)
    
    # Check if we're in the right directory
    if not os.path.exists("openAI"):
        print("❌ Error: 'openAI' directory not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Initialize test runner
    test_runner = OpenAITestRunner()
    
    # Run all tests
    results = test_runner.run_all_tests()
    
    # Generate and display summary
    summary = test_runner.generate_summary_report()
    print(summary)
    
    # Save report
    report_file = test_runner.save_report()
    
    # Exit with appropriate code
    failed_count = sum(1 for result in results.values() if not result['success'])
    if failed_count > 0:
        print(f"\n⚠️  {failed_count} script(s) failed. Check the report for details.")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()

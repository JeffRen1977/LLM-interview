#!/usr/bin/env python3
"""
Script to extract and run the model optimization code from openAI_optimize_inference_model.md
This script demonstrates efficient inference optimization techniques including quantization.
"""

import re
import os
import sys
import subprocess
import tempfile
from pathlib import Path

def extract_code_from_markdown(markdown_file):
    """Extract Python code blocks from markdown file."""
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all Python code blocks
    code_pattern = r'```python\n(.*?)\n```'
    code_blocks = re.findall(code_pattern, content, re.DOTALL)
    
    if not code_blocks:
        # Try alternative pattern without language specification
        code_pattern = r'```\n(.*?)\n```'
        code_blocks = re.findall(code_pattern, content, re.DOTALL)
    
    # Also look for inline code that starts with import
    inline_pattern = r'^import.*?$'
    inline_code = re.findall(inline_pattern, content, re.MULTILINE)
    
    return code_blocks, inline_code

def create_optimization_script():
    """Create the complete optimization script based on the markdown content."""
    
    script_content = '''#!/usr/bin/env python3
"""
Model Optimization Demo: Efficient Inference for Large Models
Extracted from openAI_optimize_inference_model.md

This script demonstrates:
1. Model quantization techniques
2. Performance comparison between FP32 and INT8 models
3. Memory and speed optimization
"""

import torch
import torch.quantization
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    print("🚀 Starting Model Optimization Demo")
    print("=" * 50)
    
    try:
        # 1. Prepare model and data
        print("📦 Loading model and preparing data...")
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load original FP32 model
        fp32_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        fp32_model.eval()  # Switch to evaluation mode
        
        # Prepare sample text
        text = "This is a great library and I love using it!"
        inputs = tokenizer(text, return_tensors="pt")
        
        print(f"✅ Model loaded: {model_name}")
        print(f"📝 Sample text: '{text}'")
        
        # 2. Evaluate original FP32 model performance
        print("\\n🔍 Evaluating original FP32 model...")
        
        # Measure model size
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            torch.save(fp32_model.state_dict(), tmp_file.name)
            fp32_size = os.path.getsize(tmp_file.name) / (1024 * 1024)
            os.unlink(tmp_file.name)
        
        print(f"📊 Original FP32 model size: {fp32_size:.2f} MB")
        
        # Measure inference latency
        print("⏱️  Measuring inference latency...")
        with torch.no_grad():
            start_time = time.time()
            for _ in range(100):
                _ = fp32_model(**inputs)
            end_time = time.time()
        
        fp32_latency = (end_time - start_time) * 10  # ms per inference (1000/100)
        print(f"⚡ Original FP32 model average latency: {fp32_latency:.2f} ms")
        
        # 3. Apply dynamic quantization (convert to INT8)
        print("\\n🔧 Applying dynamic quantization...")
        
        # This is the simplest PTQ, mainly quantizing linear layers and RNN layers
        # `torch.quantization.quantize_dynamic` automatically handles everything
        quantized_model = torch.quantization.quantize_dynamic(
            fp32_model,
            {torch.nn.Linear},  # Specify module types to quantize
            dtype=torch.qint8   # Specify quantized data type
        )
        quantized_model.eval()
        
        print("✅ Dynamic quantization completed")
        
        # 4. Evaluate quantized INT8 model performance
        print("\\n🔍 Evaluating quantized INT8 model...")
        
        # Measure model size
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            torch.save(quantized_model.state_dict(), tmp_file.name)
            quantized_size = os.path.getsize(tmp_file.name) / (1024 * 1024)
            os.unlink(tmp_file.name)
        
        print(f"📊 Quantized INT8 model size: {quantized_size:.2f} MB")
        
        # Measure inference latency (Note: dynamic quantization shows best acceleration on CPU)
        print("⏱️  Measuring quantized model latency...")
        with torch.no_grad():
            start_time = time.time()
            for _ in range(100):
                _ = quantized_model(**inputs)
            end_time = time.time()
        
        quantized_latency = (end_time - start_time) * 10  # ms per inference
        print(f"⚡ Quantized INT8 model average latency: {quantized_latency:.2f} ms")
        
        # 5. Print optimization results
        print("\\n" + "=" * 50)
        print("📈 OPTIMIZATION RESULTS SUMMARY")
        print("=" * 50)
        
        compression_ratio = fp32_size / quantized_size
        speedup_ratio = fp32_latency / quantized_latency
        
        print(f"🗜️  Model size compression ratio: {compression_ratio:.2f}x")
        print(f"🚀 Inference speedup ratio: {speedup_ratio:.2f}x")
        print(f"💾 Memory saved: {fp32_size - quantized_size:.2f} MB ({(1 - quantized_size/fp32_size)*100:.1f}%)")
        print(f"⚡ Time saved per inference: {fp32_latency - quantized_latency:.2f} ms ({(1 - quantized_latency/fp32_latency)*100:.1f}%)")
        
        # Additional analysis
        print("\\n📋 Additional Analysis:")
        print(f"   • Original model: {fp32_size:.2f} MB, {fp32_latency:.2f} ms")
        print(f"   • Optimized model: {quantized_size:.2f} MB, {quantized_latency:.2f} ms")
        
        if compression_ratio > 2.0:
            print("   ✅ Excellent compression achieved!")
        if speedup_ratio > 1.5:
            print("   ✅ Significant speedup achieved!")
        
        print("\\n🎉 Model optimization demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during optimization: {str(e)}")
        print("\\n💡 Troubleshooting tips:")
        print("   • Make sure you have internet connection for model download")
        print("   • Check if transformers and torch are properly installed")
        print("   • Try running: pip install torch transformers")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    return script_content

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['torch', 'transformers']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(packages):
    """Install missing dependencies."""
    if not packages:
        return True
    
    print(f"📦 Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def main():
    """Main function to extract code and run optimization demo."""
    print("🔍 Model Optimization Code Extractor and Runner")
    print("=" * 60)
    
    # Check if markdown file exists
    markdown_file = "openAI_optimize_inference_model.md"
    if not os.path.exists(markdown_file):
        print(f"❌ Markdown file not found: {markdown_file}")
        return False
    
    print(f"📄 Found markdown file: {markdown_file}")
    
    # Extract code from markdown
    print("🔍 Extracting code from markdown...")
    code_blocks, inline_code = extract_code_from_markdown(markdown_file)
    
    if code_blocks:
        print(f"✅ Found {len(code_blocks)} code block(s)")
    if inline_code:
        print(f"✅ Found {len(inline_code)} inline code line(s)")
    
    # Check dependencies
    print("\\n🔍 Checking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        if not install_dependencies(missing_packages):
            print("❌ Failed to install dependencies. Please install manually:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
    else:
        print("✅ All dependencies are available")
    
    # Create and run optimization script
    print("\\n📝 Creating optimization script...")
    script_content = create_optimization_script()
    
    script_file = "model_optimization_demo.py"
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ Created script: {script_file}")
    
    # Run the optimization demo
    print("\\n🚀 Running model optimization demo...")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, script_file], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        print("\\n✅ Demo completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Demo failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\\n⏹️  Demo interrupted by user")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\\n🎉 All done! Check the results above.")
    else:
        print("\\n💥 Something went wrong. Check the error messages above.")
    
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
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
import tempfile
import sys

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
        print("\n🔍 Evaluating original FP32 model...")
        
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
        print("\n🔧 Applying dynamic quantization...")
        
        try:
            # This is the simplest PTQ, mainly quantizing linear layers and RNN layers
            # `torch.quantization.quantize_dynamic` automatically handles everything
            quantized_model = torch.quantization.quantize_dynamic(
                fp32_model,
                {torch.nn.Linear},  # Specify module types to quantize
                dtype=torch.qint8   # Specify quantized data type
            )
            quantized_model.eval()
            print("✅ Dynamic quantization completed")
            
        except Exception as e:
            print(f"⚠️  Dynamic quantization failed: {str(e)}")
            print("🔄 Trying alternative: FP16 conversion...")
            
            # Alternative: Convert to FP16 (half precision)
            quantized_model = fp32_model.half()
            quantized_model.eval()
            print("✅ FP16 conversion completed (alternative to INT8 quantization)")
        
        # 4. Evaluate quantized INT8 model performance
        print("\n🔍 Evaluating quantized INT8 model...")
        
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
                # Handle FP16 inputs if needed
                if hasattr(quantized_model, 'dtype') and quantized_model.dtype == torch.float16:
                    # Convert inputs to FP16 for FP16 model
                    fp16_inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                    _ = quantized_model(**fp16_inputs)
                else:
                    _ = quantized_model(**inputs)
            end_time = time.time()
        
        quantized_latency = (end_time - start_time) * 10  # ms per inference
        print(f"⚡ Quantized INT8 model average latency: {quantized_latency:.2f} ms")
        
        # 5. Print optimization results
        print("\n" + "=" * 50)
        print("📈 OPTIMIZATION RESULTS SUMMARY")
        print("=" * 50)
        
        compression_ratio = fp32_size / quantized_size
        speedup_ratio = fp32_latency / quantized_latency
        
        print(f"🗜️  Model size compression ratio: {compression_ratio:.2f}x")
        print(f"🚀 Inference speedup ratio: {speedup_ratio:.2f}x")
        print(f"💾 Memory saved: {fp32_size - quantized_size:.2f} MB ({(1 - quantized_size/fp32_size)*100:.1f}%)")
        print(f"⚡ Time saved per inference: {fp32_latency - quantized_latency:.2f} ms ({(1 - quantized_latency/fp32_latency)*100:.1f}%)")
        
        # Additional analysis
        print("\n📋 Additional Analysis:")
        print(f"   • Original model: {fp32_size:.2f} MB, {fp32_latency:.2f} ms")
        print(f"   • Optimized model: {quantized_size:.2f} MB, {quantized_latency:.2f} ms")
        
        if compression_ratio > 2.0:
            print("   ✅ Excellent compression achieved!")
        if speedup_ratio > 1.5:
            print("   ✅ Significant speedup achieved!")
        
        print("\n🎉 Model optimization demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during optimization: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("   • Make sure you have internet connection for model download")
        print("   • Check if transformers and torch are properly installed")
        print("   • Try running: pip install torch transformers")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

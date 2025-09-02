#!/usr/bin/env python3
"""
Simple script to run the model optimization code directly from the markdown file.
This is a streamlined version that focuses on the core optimization demonstration.
"""

import torch
import torch.quantization
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
import os
import tempfile

def main():
    print("🚀 Model Optimization Demo - Simple Version")
    print("=" * 50)
    
    # 1. Prepare model and data
    print("📦 Loading model and preparing data...")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load original FP32 model
    fp32_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    fp32_model.eval()
    
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
    with torch.no_grad():
        start_time = time.time()
        for _ in range(100):
            _ = fp32_model(**inputs)
        end_time = time.time()
    
    fp32_latency = (end_time - start_time) * 10  # ms per inference
    print(f"⚡ Original FP32 model average latency: {fp32_latency:.2f} ms")
    
    # 3. Apply optimization (FP16 conversion as fallback)
    print("\n🔧 Applying model optimization...")
    
    # Try INT8 quantization first, fallback to FP16
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            fp32_model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        optimization_type = "INT8 Quantization"
    except Exception:
        print("⚠️  INT8 quantization not available, using FP16 conversion")
        quantized_model = fp32_model.half()
        optimization_type = "FP16 Conversion"
    
    quantized_model.eval()
    print(f"✅ {optimization_type} completed")
    
    # 4. Evaluate optimized model performance
    print(f"\n🔍 Evaluating optimized model...")
    
    # Measure model size
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
        torch.save(quantized_model.state_dict(), tmp_file.name)
        quantized_size = os.path.getsize(tmp_file.name) / (1024 * 1024)
        os.unlink(tmp_file.name)
    
    print(f"📊 Optimized model size: {quantized_size:.2f} MB")
    
    # Measure inference latency
    with torch.no_grad():
        start_time = time.time()
        for _ in range(100):
            # Handle FP16 inputs if needed
            if hasattr(quantized_model, 'dtype') and quantized_model.dtype == torch.float16:
                fp16_inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                _ = quantized_model(**fp16_inputs)
            else:
                _ = quantized_model(**inputs)
        end_time = time.time()
    
    quantized_latency = (end_time - start_time) * 10  # ms per inference
    print(f"⚡ Optimized model average latency: {quantized_latency:.2f} ms")
    
    # 5. Print results
    print("\n" + "=" * 50)
    print("📈 OPTIMIZATION RESULTS")
    print("=" * 50)
    
    compression_ratio = fp32_size / quantized_size
    speedup_ratio = fp32_latency / quantized_latency
    
    print(f"🗜️  Model size compression: {compression_ratio:.2f}x")
    print(f"🚀 Inference speedup: {speedup_ratio:.2f}x")
    print(f"💾 Memory saved: {fp32_size - quantized_size:.2f} MB")
    print(f"⚡ Time saved per inference: {fp32_latency - quantized_latency:.2f} ms")
    
    print(f"\n🎯 Optimization method used: {optimization_type}")
    print("🎉 Demo completed successfully!")

if __name__ == "__main__":
    main()

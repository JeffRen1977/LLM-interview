# Model Optimization Demo

This directory contains scripts to demonstrate efficient inference optimization techniques for large language models, extracted from `openAI_optimize_inference_model.md`.

## Files Created

1. **`extract_and_run_optimization.py`** - Main script that extracts code from markdown and runs the demo
2. **`model_optimization_demo.py`** - Generated optimization demo script with comprehensive features
3. **`run_optimization_simple.py`** - Simplified version for direct execution
4. **`README_optimization.md`** - This documentation file

## Quick Start

### Option 1: Run the Complete Demo (Recommended)
```bash
# Activate virtual environment
source env/bin/activate

# Run the main extraction and demo script
python extract_and_run_optimization.py
```

### Option 2: Run the Simple Demo
```bash
# Activate virtual environment
source env/bin/activate

# Run the simple optimization demo
python run_optimization_simple.py
```

### Option 3: Run the Generated Demo Directly
```bash
# Activate virtual environment
source env/bin/activate

# Run the generated demo script
python model_optimization_demo.py
```

## What the Demo Shows

The optimization demo demonstrates:

1. **Model Loading**: Downloads and loads a pre-trained DistilBERT model
2. **Performance Baseline**: Measures original FP32 model size and inference latency
3. **Model Optimization**: Applies quantization techniques (INT8 or FP16 fallback)
4. **Performance Comparison**: Shows improvements in model size and inference speed
5. **Results Analysis**: Provides detailed metrics and optimization ratios

## Expected Results

Based on the demo runs, you should see results similar to:

```
📊 Original FP32 model size: ~255 MB
⚡ Original FP32 model average latency: ~14 ms

📊 Optimized model size: ~128 MB  
⚡ Optimized model average latency: ~6-7 ms

🗜️  Model size compression: 2.00x
🚀 Inference speedup: 2.0-2.2x
💾 Memory saved: ~128 MB (50%)
⚡ Time saved per inference: ~7 ms (50%)
```

## Optimization Techniques Demonstrated

### 1. Model Quantization
- **INT8 Quantization**: Reduces model precision from 32-bit to 8-bit integers
- **FP16 Conversion**: Fallback to 16-bit floating point when INT8 is not available
- **Benefits**: Significant reduction in model size and memory usage

### 2. Performance Metrics
- **Model Size**: Measured in MB by saving model state
- **Inference Latency**: Average time per inference over 100 runs
- **Compression Ratio**: Size reduction factor
- **Speedup Ratio**: Performance improvement factor

## Technical Details

### Dependencies
- `torch` - PyTorch framework
- `transformers` - Hugging Face transformers library
- `tempfile` - For temporary file handling
- `time` - For performance measurement

### Model Used
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Task**: Sentiment analysis
- **Size**: ~255 MB (FP32)
- **Architecture**: DistilBERT (distilled BERT)

### Optimization Methods
1. **Primary**: Dynamic Post-Training Quantization (PTQ) to INT8
2. **Fallback**: FP16 conversion when quantization engine unavailable
3. **Target Layers**: Linear layers (most computationally expensive)

## Troubleshooting

### Common Issues

1. **Quantization Engine Not Available**
   ```
   ⚠️  Dynamic quantization failed: Didn't find engine for operation quantized::linear_prepack NoQEngine
   ```
   - **Solution**: The script automatically falls back to FP16 conversion
   - **Note**: This is normal on some PyTorch installations

2. **Model Download Issues**
   ```
   ❌ Error during optimization: [Connection/Download Error]
   ```
   - **Solution**: Check internet connection
   - **Alternative**: Ensure transformers library is properly installed

3. **Memory Issues**
   - **Solution**: The demo uses a relatively small model (DistilBERT)
   - **Note**: For larger models, consider using GPU if available

### Performance Notes

- **CPU vs GPU**: The demo runs on CPU by default
- **Quantization Benefits**: Most effective on CPU, less on modern GPUs
- **Model Size**: Actual savings may vary based on model architecture
- **Latency**: Results depend on hardware and system load

## Educational Value

This demo illustrates key concepts in model optimization:

1. **Profiling**: Measuring baseline performance before optimization
2. **Quantization**: Reducing model precision for efficiency
3. **Trade-offs**: Balancing model size, speed, and accuracy
4. **Fallback Strategies**: Handling different optimization scenarios
5. **Performance Analysis**: Comprehensive metrics and reporting

## Next Steps

To extend this demo, consider:

1. **GPU Acceleration**: Modify scripts to use CUDA if available
2. **Different Models**: Try with other transformer models
3. **Advanced Quantization**: Implement QAT (Quantization-Aware Training)
4. **Batch Processing**: Add batch inference optimization
5. **Memory Profiling**: Add detailed memory usage analysis

## References

- Original content from `openAI_optimize_inference_model.md`
- PyTorch Quantization Documentation
- Hugging Face Transformers Library
- Model optimization best practices

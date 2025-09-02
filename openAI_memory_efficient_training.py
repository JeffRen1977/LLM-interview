#!/usr/bin/env python3
"""
Code Extractor for Markdown Files

This script extracts Python code blocks from markdown files and saves them
as separate Python files. It's particularly useful for extracting code from
technical documentation or interview preparation materials.

Author: Generated for LLM Interview Preparation
"""

import re
import os
import argparse
from pathlib import Path


class MarkdownCodeExtractor:
    """Extract code blocks from markdown files"""
    
    def __init__(self, input_file, output_dir="extracted_code"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_code_blocks(self):
        """Extract all code blocks from the markdown file"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all code blocks
        code_blocks = re.findall(r'```(\w+)?\n(.*?)```', content, re.DOTALL)
        
        extracted_files = []
        
        for i, (language, code) in enumerate(code_blocks):
            # Skip non-Python code blocks
            if language and language.lower() not in ['python', 'py']:
                continue
                
            # Clean up the code
            code = code.strip()
            if not code:
                continue
            
            # Generate filename
            if language:
                filename = f"extracted_code_{i+1}.py"
            else:
                filename = f"extracted_code_{i+1}.py"
            
            output_path = self.output_dir / filename
            
            # Write the code to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            extracted_files.append(output_path)
            print(f"✅ Extracted code block {i+1} to: {output_path}")
        
        return extracted_files
    
    def extract_specific_sections(self, section_patterns):
        """Extract specific sections based on patterns"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        extracted_sections = []
        
        for pattern_name, pattern in section_patterns.items():
            matches = re.findall(pattern, content, re.DOTALL)
            
            for i, match in enumerate(matches):
                filename = f"{pattern_name}_{i+1}.py"
                output_path = self.output_dir / filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(match.strip())
                
                extracted_sections.append(output_path)
                print(f"✅ Extracted {pattern_name} section {i+1} to: {output_path}")
        
        return extracted_sections


def extract_memory_efficient_training_code():
    """Extract code specifically from the memory efficient training markdown"""
    
    # Define patterns for different code sections
    section_patterns = {
        'memory_intensive_block': r'class MemoryIntensiveBlock\(nn\.Module\):.*?(?=\nclass|\n# ---|\n```|\Z)',
        'large_model': r'class LargeModel\(nn\.Module\):.*?(?=\n# ---|\n```|\Z)',
        'training_step': r'def run_training_step\(.*?(?=\n# ---|\n```|\Z)',
        'main_experiment': r'# --- 2\. 实验对比 ---.*?(?=\n```|\Z)',
        'imports': r'import torch.*?(?=\n# ---|\nclass|\n```)',
    }
    
    extractor = MarkdownCodeExtractor('openAI_memory_efficient_training.md')
    
    print("🔍 Extracting code from openAI_memory_efficient_training.md...")
    print("=" * 60)
    
    # Extract all code blocks
    all_blocks = extractor.extract_code_blocks()
    
    # Extract specific sections
    specific_sections = extractor.extract_specific_sections(section_patterns)
    
    # Create a complete implementation file
    create_complete_implementation()
    
    print("\n" + "=" * 60)
    print(f"🎉 Extraction complete! Found {len(all_blocks)} code blocks")
    print(f"📁 Output directory: {extractor.output_dir}")
    print("=" * 60)


def create_complete_implementation():
    """Create a complete, runnable implementation file"""
    
    complete_code = '''#!/usr/bin/env python3
"""
Memory-Efficient Training Implementation

Complete implementation extracted from openAI_memory_efficient_training.md
This includes mixed precision training, gradient checkpointing, and memory monitoring.

Author: Extracted from LLM Interview Preparation
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import GradScaler, autocast
import time


class MemoryIntensiveBlock(nn.Module):
    """A memory-intensive block for demonstration purposes"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4 * hidden_dim, hidden_dim)
    
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class LargeModel(nn.Module):
    """A large model that can use gradient checkpointing"""
    
    def __init__(self, num_layers=32, hidden_dim=1024, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.layers = nn.ModuleList(
            [MemoryIntensiveBlock(hidden_dim) for _ in range(num_layers)]
        )
    
    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                # Only use checkpoint during training
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        return x


def run_training_step(model, optimizer, use_amp=False):
    """Run a training step and report peak memory usage"""
    scaler = GradScaler() if use_amp else None
    
    # Simulate input data
    input_data = torch.randn(16, 1024, 1024).cuda()  # (batch, seq_len, hidden)
    model.train()
    optimizer.zero_grad()
    
    # Core: Mixed precision training
    # autocast automatically converts operations to FP16
    with autocast(enabled=use_amp):
        output = model(input_data)
        loss = output.mean()

    if use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:  # Standard FP32 training
        loss.backward()
        optimizer.step()
        
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    torch.cuda.reset_peak_memory_stats()  # Reset statistics
    return peak_memory_gb


def compare_memory_usage():
    """Compare memory usage across different optimization techniques"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("❌ CUDA GPU required to run this example.")
        return
    
    print("🚀 Memory-Efficient Training Comparison")
    print("=" * 60)
    
    results = {}
    
    # Scenario 1: Standard training (might OOM)
    print("--- Scenario 1: Standard FP32 Training ---")
    model_base = LargeModel(use_checkpointing=False).to(device)
    optimizer_base = torch.optim.Adam(model_base.parameters())
    try:
        mem_base = run_training_step(model_base, optimizer_base, use_amp=False)
        results['Standard FP32'] = mem_base
        print(f"✅ Peak memory usage: {mem_base:.2f} GB")
    except RuntimeError as e:
        results['Standard FP32'] = "OOM"
        print(f"❌ Out of Memory: {e}")
    del model_base, optimizer_base
    torch.cuda.empty_cache()

    # Scenario 2: Mixed precision only
    print("\\n--- Scenario 2: Mixed Precision (AMP) ---")
    model_amp = LargeModel(use_checkpointing=False).to(device)
    optimizer_amp = torch.optim.Adam(model_amp.parameters())
    mem_amp = run_training_step(model_amp, optimizer_amp, use_amp=True)
    results['Mixed Precision'] = mem_amp
    print(f"✅ Peak memory usage: {mem_amp:.2f} GB")
    del model_amp, optimizer_amp
    torch.cuda.empty_cache()

    # Scenario 3: Gradient checkpointing only
    print("\\n--- Scenario 3: Gradient Checkpointing ---")
    model_cp = LargeModel(use_checkpointing=True).to(device)
    optimizer_cp = torch.optim.Adam(model_cp.parameters())
    mem_cp = run_training_step(model_cp, optimizer_cp, use_amp=False)
    results['Gradient Checkpointing'] = mem_cp
    print(f"✅ Peak memory usage: {mem_cp:.2f} GB")
    del model_cp, optimizer_cp
    torch.cuda.empty_cache()

    # Scenario 4: Combined approach
    print("\\n--- Scenario 4: Mixed Precision + Gradient Checkpointing ---")
    model_both = LargeModel(use_checkpointing=True).to(device)
    optimizer_both = torch.optim.Adam(model_both.parameters())
    mem_both = run_training_step(model_both, optimizer_both, use_amp=True)
    results['Combined'] = mem_both
    print(f"✅ Peak memory usage: {mem_both:.2f} GB")
    del model_both, optimizer_both
    torch.cuda.empty_cache()

    # Print summary
    print("\\n" + "=" * 60)
    print("📊 Memory Usage Summary")
    print("=" * 60)
    for technique, memory in results.items():
        if memory == "OOM":
            print(f"{technique:25} | OOM (Out of Memory)")
        else:
            print(f"{technique:25} | {memory:.2f} GB")
    
    return results


def benchmark_performance():
    """Benchmark training speed with different techniques"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("❌ CUDA GPU required for performance benchmarking.")
        return
    
    print("\\n🏃 Performance Benchmarking")
    print("=" * 60)
    
    num_iterations = 10
    results = {}
    
    # Test different configurations
    configs = [
        ("Standard FP32", False, False),
        ("Mixed Precision", True, False),
        ("Gradient Checkpointing", False, True),
        ("Combined", True, True),
    ]
    
    for config_name, use_amp, use_checkpointing in configs:
        print(f"\\nTesting {config_name}...")
        
        model = LargeModel(use_checkpointing=use_checkpointing).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Warmup
        for _ in range(3):
            run_training_step(model, optimizer, use_amp=use_amp)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            run_training_step(model, optimizer, use_amp=use_amp)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        results[config_name] = avg_time
        
        print(f"  Average time per iteration: {avg_time:.4f}s")
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    # Print performance summary
    print("\\n" + "=" * 60)
    print("⏱️  Performance Summary")
    print("=" * 60)
    baseline_time = results.get("Standard FP32", 1.0)
    
    for technique, time_taken in results.items():
        speedup = baseline_time / time_taken if time_taken > 0 else 0
        print(f"{technique:25} | {time_taken:.4f}s | {speedup:.2f}x speedup")
    
    return results


def main():
    """Main function to run all demonstrations"""
    
    print("🧠 Memory-Efficient Training Implementation")
    print("Extracted from openAI_memory_efficient_training.md")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"   Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA not available. Some features may not work.")
    
    # Run memory comparison
    memory_results = compare_memory_usage()
    
    # Run performance benchmark
    performance_results = benchmark_performance()
    
    print("\\n" + "=" * 60)
    print("🎉 All demonstrations completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
'''
    
    output_path = Path("extracted_code") / "memory_efficient_training_complete.py"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(complete_code)
    
    print(f"✅ Created complete implementation: {output_path}")


def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description="Extract code from markdown files")
    parser.add_argument("input_file", nargs="?", default="openAI_memory_efficient_training.md",
                       help="Input markdown file (default: openAI_memory_efficient_training.md)")
    parser.add_argument("-o", "--output", default="extracted_code",
                       help="Output directory (default: extracted_code)")
    parser.add_argument("--memory-training", action="store_true",
                       help="Extract memory efficient training code specifically")
    
    args = parser.parse_args()
    
    if args.memory_training:
        extract_memory_efficient_training_code()
    else:
        extractor = MarkdownCodeExtractor(args.input_file, args.output)
        extracted_files = extractor.extract_code_blocks()
        
        print(f"\\n🎉 Extraction complete! Found {len(extracted_files)} code blocks")
        print(f"📁 Output directory: {extractor.output_dir}")


if __name__ == "__main__":
    main()

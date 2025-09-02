#!/usr/bin/env python3
"""
ML Pipeline Optimization Demo
Extracted from openAI_optimize_pipeline.md

This script demonstrates comprehensive ML pipeline optimization techniques including:
1. Performance profiling and analysis
2. Data loading and memory optimization
3. Feature engineering optimization
4. Model training optimization
5. Inference optimization
6. Performance comparison between optimized and basic pipelines
"""

import numpy as np
import pandas as pd
import time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import cProfile
import io
import pstats
import warnings
import tempfile
import os
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class OptimizedMLPipeline:
    """优化后的机器学习流水线"""
    
    def __init__(self, use_gpu=False, n_jobs=-1):
        self.use_gpu = use_gpu
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.pipeline = None
        self.feature_cache = {}
        
    def profile_pipeline(self, func, *args, **kwargs):
        """性能分析工具"""
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(10)  # 打印前10个最耗时的函数
        print("\n📊 Profile Results:")
        print(s.getvalue())
        
        return result
    
    def optimize_data_loading(self, file_path=None, chunk_size=10000):
        """优化数据加载 - 使用分块读取"""
        print("📦 Optimizing data loading...")
        
        if file_path:
            # 分块读取大文件
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # 并行处理每个chunk
                chunks.append(self._process_chunk(chunk))
            return pd.concat(chunks, ignore_index=True)
        else:
            # 示例数据
            print("   Generating synthetic classification data...")
            X, y = make_classification(
                n_samples=100000, 
                n_features=20, 
                n_informative=15,
                n_redundant=5,
                n_classes=2,
                random_state=42
            )
            return pd.DataFrame(X), pd.Series(y)
    
    def _process_chunk(self, chunk):
        """处理数据块"""
        # 基础数据清理
        chunk = chunk.dropna(thresh=len(chunk.columns) * 0.8)  # 删除80%以上缺失的行
        return chunk
    
    def optimize_feature_engineering(self, X, y):
        """优化特征工程"""
        start_time = time.time()
        
        # 1. 特征缓存机制
        feature_hash = hash(str(X.shape) + str(X.dtypes.to_list()))
        if feature_hash in self.feature_cache:
            print("   ✅ Using cached features")
            return self.feature_cache[feature_hash]
        
        # 2. 并行特征工程
        print("🔧 Starting feature engineering optimization...")
        
        # 数值特征和分类特征分离
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print(f"   📊 Numeric features: {len(numeric_features)}")
        print(f"   📊 Categorical features: {len(categorical_features)}")
        
        # 构建预处理pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            n_jobs=self.n_jobs
        )
        
        # 应用预处理
        X_processed = preprocessor.fit_transform(X)
        
        # 缓存结果
        self.feature_cache[feature_hash] = (X_processed, preprocessor)
        
        print(f"   ✅ Feature engineering completed in {time.time() - start_time:.2f}s")
        return X_processed, preprocessor
    
    def optimize_model_training(self, X, y):
        """优化模型训练"""
        print("🤖 Starting model training optimization...")
        start_time = time.time()
        
        # 1. 使用并行训练
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_jobs=self.n_jobs,  # 并行训练
            random_state=42
        )
        
        # 2. 早停策略 (这里用简化版本)
        # 实际中可以使用validation set进行early stopping
        
        model.fit(X, y)
        
        print(f"   ✅ Model training completed in {time.time() - start_time:.2f}s")
        return model
    
    def optimize_inference(self, model, X_test, batch_size=1000):
        """优化推理过程"""
        print("⚡ Starting batch inference optimization...")
        start_time = time.time()
        
        predictions = []
        
        # 批量推理以减少内存使用
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            batch_pred = model.predict(batch)
            predictions.extend(batch_pred)
        
        print(f"   ✅ Inference completed in {time.time() - start_time:.2f}s")
        return np.array(predictions)
    
    def memory_optimization(self, X):
        """内存优化"""
        print("💾 Starting memory optimization...")
        original_memory = X.memory_usage(deep=True).sum() / 1024**2
        
        # 优化数据类型
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].dtype == 'int64':
                if X[col].min() >= -128 and X[col].max() <= 127:
                    X[col] = X[col].astype('int8')
                elif X[col].min() >= -32768 and X[col].max() <= 32767:
                    X[col] = X[col].astype('int16')
                elif X[col].min() >= -2147483648 and X[col].max() <= 2147483647:
                    X[col] = X[col].astype('int32')
            elif X[col].dtype == 'float64':
                X[col] = pd.to_numeric(X[col], downcast='float')
        
        optimized_memory = X.memory_usage(deep=True).sum() / 1024**2
        memory_saved = ((original_memory - optimized_memory) / original_memory * 100)
        
        print(f"   ✅ Memory optimization: {original_memory:.2f}MB -> {optimized_memory:.2f}MB "
              f"(saved {memory_saved:.1f}%)")
        
        return X
    
    def gpu_acceleration_example(self, X, y):
        """GPU加速示例 (需要安装cuml)"""
        if not self.use_gpu:
            print("   ℹ️  GPU acceleration not enabled")
            return None
            
        try:
            # 这里只是示例，实际使用需要安装cuml
            # from cuml.ensemble import RandomForestClassifier as cuRF
            # model = cuRF(n_estimators=100, max_depth=10)
            # model.fit(X, y)
            print("   ℹ️  GPU acceleration example (requires cuml installation)")
        except ImportError:
            print("   ⚠️  cuml not installed, GPU acceleration unavailable")
    
    def build_optimized_pipeline(self):
        """构建完整的优化流水线"""
        print("=" * 60)
        print("🚀 Building Optimized ML Pipeline")
        print("=" * 60)
        
        total_start_time = time.time()
        
        # 1. 数据加载优化
        print("\n1️⃣  Optimizing data loading...")
        X, y = self.optimize_data_loading()
        print(f"   📊 Data shape: {X.shape}")
        
        # 2. 内存优化
        X = self.memory_optimization(X)
        
        # 3. 数据分割
        print("\n2️⃣  Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"   📊 Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # 4. 特征工程优化
        print("\n3️⃣  Optimizing feature engineering...")
        X_train_processed, preprocessor = self.optimize_feature_engineering(X_train, y_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # 5. 模型训练优化
        print("\n4️⃣  Optimizing model training...")
        model = self.optimize_model_training(X_train_processed, y_train)
        
        # 6. 推理优化
        print("\n5️⃣  Optimizing inference...")
        predictions = self.optimize_inference(model, X_test_processed)
        
        # 7. GPU加速示例
        print("\n6️⃣  GPU acceleration example...")
        self.gpu_acceleration_example(X_train_processed, y_train)
        
        # 8. 模型保存优化
        print("\n7️⃣  Saving optimized models...")
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_model:
            joblib.dump(model, tmp_model.name, compress=3)  # 压缩保存
            model_size = os.path.getsize(tmp_model.name) / 1024**2
            os.unlink(tmp_model.name)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_prep:
            joblib.dump(preprocessor, tmp_prep.name, compress=3)
            prep_size = os.path.getsize(tmp_prep.name) / 1024**2
            os.unlink(tmp_prep.name)
        
        print(f"   💾 Model size: {model_size:.2f}MB, Preprocessor size: {prep_size:.2f}MB")
        
        total_time = time.time() - total_start_time
        accuracy = (predictions == y_test).mean()
        
        print("\n" + "=" * 60)
        print("📈 PIPELINE OPTIMIZATION RESULTS")
        print("=" * 60)
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"🎯 Prediction accuracy: {accuracy:.4f}")
        print(f"💾 Total model size: {model_size + prep_size:.2f}MB")
        
        return model, preprocessor, predictions

# 性能对比示例
def compare_pipelines():
    """对比优化前后的性能"""
    print("\n" + "=" * 60)
    print("📊 Performance Comparison: Basic vs Optimized Pipeline")
    print("=" * 60)
    
    # 创建测试数据
    print("\n📦 Creating test data...")
    X, y = make_classification(n_samples=50000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 未优化的基础流水线
    print("\n1️⃣  Basic Pipeline (Unoptimized)")
    start_time = time.time()
    
    basic_model = RandomForestClassifier(n_estimators=100, random_state=42)
    basic_model.fit(X_train, y_train)
    basic_pred = basic_model.predict(X_test)
    basic_time = time.time() - start_time
    basic_accuracy = (basic_pred == y_test).mean()
    
    print(f"   ⏱️  Basic pipeline time: {basic_time:.2f}s")
    print(f"   🎯 Basic pipeline accuracy: {basic_accuracy:.4f}")
    
    # 优化后的流水线
    print("\n2️⃣  Optimized Pipeline")
    start_time = time.time()
    
    optimized_pipeline = OptimizedMLPipeline(n_jobs=mp.cpu_count())
    X_df, y_series = pd.DataFrame(X_train), pd.Series(y_train)
    X_processed, preprocessor = optimized_pipeline.optimize_feature_engineering(X_df, y_series)
    optimized_model = optimized_pipeline.optimize_model_training(X_processed, y_series)
    
    X_test_processed = preprocessor.transform(pd.DataFrame(X_test))
    optimized_pred = optimized_pipeline.optimize_inference(optimized_model, X_test_processed)
    optimized_time = time.time() - start_time
    optimized_accuracy = (optimized_pred == y_test).mean()
    
    print(f"   ⏱️  Optimized pipeline time: {optimized_time:.2f}s")
    print(f"   🎯 Optimized pipeline accuracy: {optimized_accuracy:.4f}")
    
    # 性能提升
    speedup = basic_time / optimized_time
    time_saved = ((basic_time - optimized_time) / basic_time * 100)
    
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE IMPROVEMENT SUMMARY")
    print("=" * 60)
    print(f"🚀 Speedup ratio: {speedup:.2f}x")
    print(f"⏱️  Time saved: {time_saved:.1f}%")
    print(f"🎯 Accuracy difference: {abs(optimized_accuracy - basic_accuracy):.4f}")
    
    if speedup > 1.5:
        print("   ✅ Significant performance improvement achieved!")
    if abs(optimized_accuracy - basic_accuracy) < 0.01:
        print("   ✅ Accuracy maintained while improving performance!")

def main():
    """Main function to run the pipeline optimization demo."""
    print("🚀 ML Pipeline Optimization Demo")
    print("=" * 60)
    print("This demo showcases comprehensive ML pipeline optimization techniques")
    print("including data loading, feature engineering, model training, and inference optimization.")
    print("=" * 60)
    
    try:
        # 运行优化流水线
        pipeline = OptimizedMLPipeline(use_gpu=False, n_jobs=-1)
        
        # 使用性能分析
        print("\n🔍 Running pipeline with performance profiling...")
        pipeline.profile_pipeline(pipeline.build_optimized_pipeline)
        
        # 运行性能对比
        compare_pipelines()
        
        print("\n" + "=" * 60)
        print("💡 OPTIMIZATION RECOMMENDATIONS SUMMARY")
        print("=" * 60)
        print("1. 🔄 Parallel Processing: Utilize multi-core CPU effectively")
        print("2. 💾 Memory Optimization: Optimize data types to reduce memory usage")
        print("3. 📦 Batch Processing: Avoid processing samples one by one")
        print("4. 🗄️  Feature Caching: Avoid redundant feature calculations")
        print("5. 🗜️  Model Compression: Use compressed formats for model storage")
        print("6. 🚀 GPU Acceleration: Use GPU for large-scale data processing")
        print("7. ⚡ Pipeline Parallelism: Execute different stages in parallel")
        print("8. 📊 Performance Analysis: Regularly profile and optimize bottlenecks")
        
        print("\n🎉 ML Pipeline optimization demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during pipeline optimization: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("   • Make sure all required packages are installed")
        print("   • Check if you have sufficient memory for the demo")
        print("   • Try reducing the dataset size if memory is limited")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

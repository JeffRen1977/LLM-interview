
## 问题分析

机器学习流水线性能优化需要从多个角度进行系统性分析和改进，包括数据预处理、特征工程、模型训练和推理等各个环节。

## 优化策略

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
        print("Profile Results:")
        print(s.getvalue())
        
        return result
    
    def optimize_data_loading(self, file_path=None, chunk_size=10000):
        """优化数据加载 - 使用分块读取"""
        if file_path:
            # 分块读取大文件
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # 并行处理每个chunk
                chunks.append(self._process_chunk(chunk))
            return pd.concat(chunks, ignore_index=True)
        else:
            # 示例数据
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
            print("使用缓存的特征")
            return self.feature_cache[feature_hash]
        
        # 2. 并行特征工程
        print("开始特征工程优化...")
        
        # 数值特征和分类特征分离
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
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
        
        print(f"特征工程完成，耗时: {time.time() - start_time:.2f}秒")
        return X_processed, preprocessor
    
    def optimize_model_training(self, X, y):
        """优化模型训练"""
        print("开始模型训练优化...")
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
        
        print(f"模型训练完成，耗时: {time.time() - start_time:.2f}秒")
        return model
    
    def optimize_inference(self, model, X_test, batch_size=1000):
        """优化推理过程"""
        print("开始批量推理...")
        start_time = time.time()
        
        predictions = []
        
        # 批量推理以减少内存使用
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            batch_pred = model.predict(batch)
            predictions.extend(batch_pred)
        
        print(f"推理完成，耗时: {time.time() - start_time:.2f}秒")
        return np.array(predictions)
    
    def memory_optimization(self, X):
        """内存优化"""
        print("开始内存优化...")
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
        print(f"内存优化完成: {original_memory:.2f}MB -> {optimized_memory:.2f}MB "
              f"(节省 {((original_memory - optimized_memory) / original_memory * 100):.1f}%)")
        
        return X
    
    def gpu_acceleration_example(self, X, y):
        """GPU加速示例 (需要安装cuml)"""
        if not self.use_gpu:
            print("GPU加速未启用")
            return None
            
        try:
            # 这里只是示例，实际使用需要安装cuml
            # from cuml.ensemble import RandomForestClassifier as cuRF
            # model = cuRF(n_estimators=100, max_depth=10)
            # model.fit(X, y)
            print("GPU加速模型训练 (示例 - 需要安装cuml)")
        except ImportError:
            print("cuml未安装，无法使用GPU加速")
    
    def build_optimized_pipeline(self):
        """构建完整的优化流水线"""
        print("=" * 50)
        print("构建优化的机器学习流水线")
        print("=" * 50)
        
        total_start_time = time.time()
        
        # 1. 数据加载优化
        print("\n1. 优化数据加载...")
        X, y = self.optimize_data_loading()
        print(f"数据形状: {X.shape}")
        
        # 2. 内存优化
        X = self.memory_optimization(X)
        
        # 3. 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 4. 特征工程优化
        X_train_processed, preprocessor = self.optimize_feature_engineering(X_train, y_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # 5. 模型训练优化
        model = self.optimize_model_training(X_train_processed, y_train)
        
        # 6. 推理优化
        predictions = self.optimize_inference(model, X_test_processed)
        
        # 7. GPU加速示例
        self.gpu_acceleration_example(X_train_processed, y_train)
        
        # 8. 模型保存优化
        print("\n保存优化后的模型...")
        joblib.dump(model, 'optimized_model.joblib', compress=3)  # 压缩保存
        joblib.dump(preprocessor, 'preprocessor.joblib', compress=3)
        
        total_time = time.time() - total_start_time
        print(f"\n总耗时: {total_time:.2f}秒")
        print(f"预测准确率: {(predictions == y_test).mean():.4f}")
        
        return model, preprocessor, predictions

# 性能对比示例
def compare_pipelines():
    """对比优化前后的性能"""
    print("=" * 60)
    print("性能对比：优化前 vs 优化后")
    print("=" * 60)
    
    # 创建测试数据
    X, y = make_classification(n_samples=50000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 未优化的基础流水线
    print("\n1. 基础流水线 (未优化)")
    start_time = time.time()
    
    basic_model = RandomForestClassifier(n_estimators=100, random_state=42)
    basic_model.fit(X_train, y_train)
    basic_pred = basic_model.predict(X_test)
    basic_time = time.time() - start_time
    
    print(f"基础流水线耗时: {basic_time:.2f}秒")
    
    # 优化后的流水线
    print("\n2. 优化后流水线")
    start_time = time.time()
    
    optimized_pipeline = OptimizedMLPipeline(n_jobs=mp.cpu_count())
    X_df, y_series = pd.DataFrame(X_train), pd.Series(y_train)
    X_processed, preprocessor = optimized_pipeline.optimize_feature_engineering(X_df, y_series)
    optimized_model = optimized_pipeline.optimize_model_training(X_processed, y_series)
    
    X_test_processed = preprocessor.transform(pd.DataFrame(X_test))
    optimized_pred = optimized_pipeline.optimize_inference(optimized_model, X_test_processed)
    optimized_time = time.time() - start_time
    
    print(f"优化流水线耗时: {optimized_time:.2f}秒")
    
    # 性能提升
    speedup = basic_time / optimized_time
    print(f"\n性能提升: {speedup:.2f}x")
    print(f"时间节省: {((basic_time - optimized_time) / basic_time * 100):.1f}%")

if __name__ == "__main__":
    # 运行优化流水线
    pipeline = OptimizedMLPipeline(use_gpu=False, n_jobs=-1)
    
    # 使用性能分析
    pipeline.profile_pipeline(pipeline.build_optimized_pipeline)
    
    print("\n" + "="*60)
    
    # 运行性能对比
    compare_pipelines()
    
    print("\n优化建议总结:")
    print("1. 并行处理: 充分利用多核CPU")
    print("2. 内存优化: 优化数据类型，减少内存使用")
    print("3. 批量处理: 避免逐个样本处理")
    print("4. 特征缓存: 避免重复计算")
    print("5. 模型压缩: 使用压缩格式保存模型")
    print("6. GPU加速: 对于大规模数据使用GPU")
    print("7. 流水线并行: 不同阶段并行执行")
    print("8. 性能分析: 定期分析瓶颈并优化")

## 优化策略详解

### 1. **性能分析 (Profiling)**
- 使用 `cProfile` 找出性能瓶颈
- 监控内存使用情况
- 分析各个步骤的耗时

### 2. **数据处理优化**
- **分块读取**: 处理大文件时使用 `pd.read_csv(chunksize=...)`
- **内存优化**: 优化数据类型，减少内存占用
- **并行处理**: 使用多进程/多线程处理数据

### 3. **特征工程优化**
- **特征缓存**: 避免重复计算相同的特征
- **并行特征计算**: 使用 `n_jobs` 参数
- **流水线设计**: 使用 `Pipeline` 和 `ColumnTransformer`

### 4. **模型训练优化**
- **并行训练**: RandomForest 等算法支持并行
- **早停策略**: 避免过度训练
- **增量学习**: 对于在线学习场景

### 5. **推理优化**
- **批量推理**: 批量处理而非逐个预测
- **模型量化**: 减少模型大小和推理时间
- **模型缓存**: 将频繁使用的模型保存在内存中

### 6. **硬件加速**
- **GPU 加速**: 使用 cuML、Rapids 等库
- **多核并行**: 充分利用 CPU 多核
- **分布式计算**: 使用 Dask、Ray 等框架

### 7. **存储优化**
- **模型压缩**: 使用 `joblib` 的压缩选项
- **特征存储**: 预计算并存储特征
- **缓存机制**: 实现智能缓存策略

这个优化方案可以根据具体的业务场景和数据特点进行调整，通常能带来 2-10x 的性能提升。关键是要先进行性能分析，找出真正的瓶颈，然后有针对性地进行优化。
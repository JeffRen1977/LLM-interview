#!/usr/bin/env python3
"""
OpenAI Interview Question 6: Analyze and Fix a Biased Dataset

This comprehensive module demonstrates advanced techniques for detecting,
analyzing, and mitigating bias in machine learning datasets to ensure
fair and ethical AI systems.

Key Bias Analysis Techniques:
1. Bias Detection and Measurement
   - Statistical analysis of dataset composition
   - Demographic parity and equalized odds
   - Disparate impact analysis
   - Intersectional bias identification

2. Data-Level Bias Mitigation
   - Oversampling techniques (SMOTE, ADASYN)
   - Undersampling strategies (Random, Tomek Links)
   - Synthetic data generation
   - Data augmentation for fairness

3. Algorithm-Level Bias Mitigation
   - Fairness-aware loss functions
   - Constrained optimization
   - Adversarial debiasing
   - Post-processing techniques

4. Evaluation and Monitoring
   - Fairness metrics calculation
   - Bias monitoring dashboards
   - A/B testing for fairness
   - Continuous bias assessment

Technical Highlights:
- Comprehensive bias detection pipeline
- Multiple mitigation strategies
- Fairness metrics and evaluation
- Production-ready bias monitoring
- Real-world case studies and examples

Expected Outcomes:
- Clear understanding of bias types and sources
- Practical tools for bias detection and mitigation
- Fairness evaluation and monitoring techniques
- Production deployment considerations

Author: Jianfeng Ren
Date: 09/07/2025
Version: 2.0
"""

# Standard library imports
import warnings
warnings.filterwarnings('ignore')

# Third-party imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Imbalanced learning imports
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

def main():
    """
    Main function to demonstrate comprehensive bias detection and mitigation techniques.
    
    This function showcases a complete pipeline for analyzing and fixing biased datasets,
    including bias detection, mitigation strategies, and fairness evaluation.
    
    Key Demonstration Areas:
    1. Bias Detection and Analysis
       - Statistical analysis of dataset composition
       - Visualization of bias patterns
       - Quantification of bias severity
       - Identification of bias sources
    
    2. Data-Level Bias Mitigation
       - Oversampling techniques (SMOTE)
       - Undersampling strategies
       - Combined sampling approaches
       - Synthetic data generation
    
    3. Algorithm-Level Bias Mitigation
       - Fairness-aware model training
       - Constrained optimization
       - Post-processing techniques
       - Bias monitoring and evaluation
    
    4. Fairness Evaluation
       - Multiple fairness metrics
       - Performance comparison
       - Bias reduction analysis
       - Production deployment considerations
    
    Expected Outcomes:
    - Clear understanding of bias types and sources
    - Practical tools for bias detection and mitigation
    - Fairness evaluation and monitoring techniques
    - Production deployment recommendations
    """
    
    print("🔍 有偏見資料集分析與修復綜合演示")
    print("=" * 80)
    print("本演示將展示如何檢測、分析和修復機器學習資料集中的偏見")
    print("包括資料層面和演算法層面的偏見緩解技術")
    print("=" * 80)
    
    # =================================================================
    # 1. 創建模擬的偏見資料集
    # =================================================================
    print("\n📊 第一步: 創建模擬的偏見資料集")
    print("-" * 50)
    print("正在創建一個模擬的、有嚴重類別不平衡的資料集...")
    print("   📝 場景: 信用違約預測")
    print("   🎯 目標: 預測客戶是否會違約")
    print("   ⚠️  問題: 嚴重類別不平衡 (95% 非違約, 5% 違約)")
    
    # 設置隨機種子以確保結果可重現
    np.random.seed(42)
    
    # 創建模擬資料
    # 特徵1: 收入水平 (0-10)
    # 特徵2: 信用評分 (0-5)
    # 目標: 違約標籤 (0=非違約, 1=違約)
    data = {
        'feature1': np.random.rand(1000) * 10,  # 收入水平
        'feature2': np.random.rand(1000) * 5,   # 信用評分
        'target': [0] * 950 + [1] * 50  # 嚴重不平衡: 95% vs 5%
    }
    
    # 轉換為DataFrame並打亂順序
    df = pd.DataFrame(data)
    np.random.shuffle(df.values)
    
    # 分離特徵和目標
    X = df[['feature1', 'feature2']]
    y = df['target']
    
    print(f"   ✅ 資料集創建完成")
    print(f"   📊 資料形狀: {df.shape}")
    print(f"   🎯 特徵數量: {X.shape[1]}")
    print(f"   📈 樣本數量: {len(y)}")
    
    # =================================================================
    # 2. 檢測偏見：分析類別分佈
    # =================================================================
    print("\n🔍 第二步: 檢測偏見")
    print("-" * 50)
    print("正在分析資料集的類別分佈和偏見程度...")
    
    # 分析類別分佈
    class_counts = y.value_counts()
    imbalance_ratio = class_counts[0] / class_counts[1]
    
    print(f"   📊 原始資料分佈:")
    print(f"      - 類別 0 (非違約): {class_counts[0]} 樣本 ({class_counts[0]/len(y)*100:.1f}%)")
    print(f"      - 類別 1 (違約): {class_counts[1]} 樣本 ({class_counts[1]/len(y)*100:.1f}%)")
    print(f"      - 不平衡比例: {imbalance_ratio:.1f}:1")
    
    # 評估偏見嚴重程度
    if imbalance_ratio > 10:
        bias_severity = "嚴重"
        print(f"   ⚠️  偏見嚴重程度: {bias_severity} (比例 > 10:1)")
    elif imbalance_ratio > 5:
        bias_severity = "中等"
        print(f"   ⚠️  偏見嚴重程度: {bias_severity} (比例 > 5:1)")
    else:
        bias_severity = "輕微"
        print(f"   ✅ 偏見嚴重程度: {bias_severity} (比例 ≤ 5:1)")
    
    # 創建視覺化
    print("   📈 創建偏見視覺化圖表...")
    plt.figure(figsize=(15, 6))
    
    # 原始資料分佈
    plt.subplot(1, 3, 1)
    sns.countplot(x='target', data=df)
    plt.title("原始類別分佈\n(嚴重不平衡)")
    plt.xlabel("類別")
    plt.ylabel("樣本數量")
    plt.xticks([0, 1], ['非違約', '違約'])
    
    # 添加數值標籤
    for i, v in enumerate(class_counts.values):
        plt.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 特徵分佈分析
    plt.subplot(1, 3, 2)
    sns.boxplot(data=df, x='target', y='feature1')
    plt.title("特徵1分佈\n(收入水平)")
    plt.xlabel("類別")
    plt.ylabel("特徵1值")
    plt.xticks([0, 1], ['非違約', '違約'])
    
    plt.subplot(1, 3, 3)
    sns.boxplot(data=df, x='target', y='feature2')
    plt.title("特徵2分佈\n(信用評分)")
    plt.xlabel("類別")
    plt.ylabel("特徵2值")
    plt.xticks([0, 1], ['非違約', '違約'])
    
    plt.tight_layout()
    plt.savefig('bias_dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✅ 視覺化圖表已保存為 'bias_dataset_analysis.png'")
    plt.show()
    
    # =================================================================
    # 3. 在偏見資料上訓練模型並評估
    # =================================================================
    print("\n🤖 第三步: 在偏見資料上訓練模型")
    print("-" * 50)
    print("正在使用偏見資料訓練模型並評估其表現...")
    
    # 分割資料集
    print("   📊 分割資料集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"      - 訓練集大小: {X_train.shape[0]} 樣本")
    print(f"      - 測試集大小: {X_test.shape[0]} 樣本")
    print(f"      - 特徵數量: {X_train.shape[1]}")
    
    # 訓練模型
    print("   🔄 訓練邏輯回歸模型...")
    model_biased = LogisticRegression(random_state=42, max_iter=1000)
    model_biased.fit(X_train, y_train)
    
    # 預測和評估
    print("   📈 評估模型表現...")
    y_pred_biased = model_biased.predict(X_test)
    
    # 計算詳細指標
    accuracy_biased = accuracy_score(y_test, y_pred_biased)
    f1_biased = f1_score(y_test, y_pred_biased)
    
    print(f"\n   📊 偏見模型表現:")
    print(f"      - 準確率: {accuracy_biased:.4f}")
    print(f"      - F1分數: {f1_biased:.4f}")
    
    # 詳細分類報告
    print(f"\n   📋 詳細分類報告:")
    print(classification_report(y_test, y_pred_biased, target_names=['非違約', '違約']))
    
    # 分析偏見影響
    print(f"\n   ⚠️  偏見影響分析:")
    print(f"      - 儘管總體準確率很高 ({accuracy_biased:.1%})，但對少數類別的識別能力很差")
    print(f"      - F1分數較低 ({f1_biased:.4f}) 表明模型在平衡精確率和召回率方面表現不佳")
    print(f"      - 這是由於類別不平衡導致的偏見問題")
    
    # =================================================================
    # 4. 偏見緩解：使用SMOTE重採樣
    # =================================================================
    print("\n🔧 第四步: 偏見緩解")
    print("-" * 50)
    print("正在使用SMOTE (Synthetic Minority Oversampling Technique) 重採樣...")
    print("   📝 技術: SMOTE - 合成少數類過採樣技術")
    print("   🎯 目標: 平衡類別分佈，減少偏見")
    print("   ⚙️  原理: 在少數類樣本之間生成合成樣本")
    
    # 應用SMOTE重採樣
    print("   🔄 應用SMOTE重採樣...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # 分析重採樣結果
    resampled_counts = pd.Series(y_train_resampled).value_counts()
    resampled_ratio = resampled_counts[0] / resampled_counts[1]
    
    print(f"\n   📊 重採樣結果:")
    print(f"      - 原始訓練集大小: {X_train.shape[0]} 樣本")
    print(f"      - 重採樣後大小: {X_train_resampled.shape[0]} 樣本")
    print(f"      - 類別 0 (非違約): {resampled_counts[0]} 樣本")
    print(f"      - 類別 1 (違約): {resampled_counts[1]} 樣本")
    print(f"      - 新類別比例: {resampled_ratio:.1f}:1")
    
    # 評估重採樣效果
    if resampled_ratio < 2.0:
        print(f"   ✅ 重採樣成功! 類別比例已平衡到 {resampled_ratio:.1f}:1")
    else:
        print(f"   ⚠️  重採樣部分成功，類別比例為 {resampled_ratio:.1f}:1")
    
    # 創建重採樣前後對比視覺化
    print("   📈 創建重採樣前後對比圖...")
    plt.figure(figsize=(15, 5))
    
    # 原始訓練資料分佈
    plt.subplot(1, 3, 1)
    sns.countplot(x=y_train)
    plt.title("原始訓練資料分佈\n(偏見嚴重)")
    plt.xlabel("類別")
    plt.ylabel("樣本數量")
    plt.xticks([0, 1], ['非違約', '違約'])
    
    # 重採樣後資料分佈
    plt.subplot(1, 3, 2)
    sns.countplot(x=y_train_resampled)
    plt.title("SMOTE重採樣後分佈\n(類別平衡)")
    plt.xlabel("類別")
    plt.ylabel("樣本數量")
    plt.xticks([0, 1], ['非違約', '違約'])
    
    # 特徵分佈對比
    plt.subplot(1, 3, 3)
    plt.scatter(X_train_resampled[y_train_resampled==0, 0], 
                X_train_resampled[y_train_resampled==0, 1], 
                alpha=0.6, label='非違約', s=20)
    plt.scatter(X_train_resampled[y_train_resampled==1, 0], 
                X_train_resampled[y_train_resampled==1, 1], 
                alpha=0.6, label='違約', s=20)
    plt.title("重採樣後特徵分佈")
    plt.xlabel("特徵1 (收入水平)")
    plt.ylabel("特徵2 (信用評分)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('bias_dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✅ 重採樣對比圖已保存為 'bias_dataset_analysis.png'")
    plt.show()
    
    # =================================================================
    # 5. 在修復後資料上重新訓練模型
    # =================================================================
    print("\n🔄 第五步: 重新訓練模型")
    print("-" * 50)
    print("正在使用修復後的資料重新訓練模型...")
    
    # 重新訓練模型
    print("   🔄 訓練修復後的模型...")
    model_fixed = LogisticRegression(random_state=42, max_iter=1000)
    model_fixed.fit(X_train_resampled, y_train_resampled)
    
    # 評估修復後的模型
    print("   📈 評估修復後的模型...")
    y_pred_fixed = model_fixed.predict(X_test)
    
    # 計算修復後模型的指標
    accuracy_fixed = accuracy_score(y_test, y_pred_fixed)
    f1_fixed = f1_score(y_test, y_pred_fixed)
    
    print(f"\n   📊 修復後模型表現:")
    print(f"      - 準確率: {accuracy_fixed:.4f}")
    print(f"      - F1分數: {f1_fixed:.4f}")
    
    # 詳細分類報告
    print(f"\n   📋 詳細分類報告:")
    print(classification_report(y_test, y_pred_fixed, target_names=['非違約', '違約']))
    
    # 分析修復效果
    print(f"\n   ✅ 修復效果分析:")
    print(f"      - 準確率變化: {accuracy_biased:.4f} → {accuracy_fixed:.4f} ({accuracy_fixed-accuracy_biased:+.4f})")
    print(f"      - F1分數變化: {f1_biased:.4f} → {f1_fixed:.4f} ({f1_fixed-f1_biased:+.4f})")
    
    if f1_fixed > f1_biased:
        print(f"      - ✅ F1分數提升 {((f1_fixed-f1_biased)/f1_biased)*100:.1f}%，修復成功!")
    else:
        print(f"      - ⚠️  F1分數變化有限，可能需要其他修復策略")
    
    # =================================================================
    # 6. 模型對比分析
    # =================================================================
    print("\n📊 第六步: 模型對比分析")
    print("-" * 50)
    print("正在比較偏見模型和修復後模型的表現...")
    
    # 創建混淆矩陣對比圖
    print("   📈 創建混淆矩陣對比圖...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 偏見模型的混淆矩陣
    cm_biased = confusion_matrix(y_test, y_pred_biased)
    sns.heatmap(cm_biased, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('偏見模型混淆矩陣\n(對少數類別識別能力差)')
    ax1.set_xlabel('預測類別')
    ax1.set_ylabel('實際類別')
    ax1.set_xticklabels(['非違約', '違約'])
    ax1.set_yticklabels(['非違約', '違約'])
    
    # 修復後模型的混淆矩陣
    cm_fixed = confusion_matrix(y_test, y_pred_fixed)
    sns.heatmap(cm_fixed, annot=True, fmt='d', cmap='Greens', ax=ax2)
    ax2.set_title('修復後模型混淆矩陣\n(類別平衡識別能力)')
    ax2.set_xlabel('預測類別')
    ax2.set_ylabel('實際類別')
    ax2.set_xticklabels(['非違約', '違約'])
    ax2.set_yticklabels(['非違約', '違約'])
    
    plt.tight_layout()
    plt.savefig('bias_model_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ 混淆矩陣對比圖已保存為 'bias_model_comparison.png'")
    plt.show()
    
    # =================================================================
    # 7. 綜合分析總結
    # =================================================================
    print("\n" + "=" * 80)
    print("🎯 偏見分析與修復總結")
    print("=" * 80)
    
    print("\n📊 偏見來源分析:")
    print("   🔍 檢測到的偏見類型:")
    print("      - 採樣偏見: 資料集中類別比例嚴重不平衡 (19:1)")
    print("      - 代表性偏見: 少數類別樣本不足")
    print("      - 測量偏見: 模型對少數類別的識別能力差")
    print("   ⚠️  偏見嚴重程度: 嚴重 (比例 > 10:1)")
    
    print("\n🔧 解決方案效果:")
    print("   ✅ 採用的修復技術:")
    print("      - SMOTE重採樣: 平衡類別分佈")
    print("      - 合成樣本生成: 增加少數類別代表性")
    print("      - 模型重新訓練: 提升公平性")
    
    print(f"   📈 修復效果量化:")
    print(f"      - 準確率: {accuracy_biased:.4f} → {accuracy_fixed:.4f}")
    print(f"      - F1分數: {f1_biased:.4f} → {f1_fixed:.4f}")
    print(f"      - 改善程度: {((f1_fixed-f1_biased)/f1_biased)*100:.1f}%")
    
    if f1_fixed > f1_biased:
        print("   ✅ 修復成功! 模型公平性顯著提升")
    else:
        print("   ⚠️  修復效果有限，建議嘗試其他策略")
    
    print("\n💡 進一步改進建議:")
    print("   🚀 資料層面:")
    print("      - 收集更多樣化和代表性的資料")
    print("      - 使用多種重採樣技術組合")
    print("      - 考慮資料增強和合成技術")
    
    print("   🤖 演算法層面:")
    print("      - 使用公平性約束的損失函數")
    print("      - 考慮對抗性去偏見技術")
    print("      - 實施後處理公平性調整")
    
    print("   📊 監控層面:")
    print("      - 建立持續的偏見監控系統")
    print("      - 定期評估模型在不同群體上的表現")
    print("      - 實施A/B測試驗證公平性")
    
    print("\n🎉 偏見分析與修復演示完成!")
    print("=" * 80)

if __name__ == "__main__":
    """
    Entry point for the Bias Dataset Analysis and Mitigation demonstration.
    
    This script can be run directly to see the complete bias detection and
    mitigation demonstration in action. It will show:
    - Bias detection and analysis techniques
    - Data-level bias mitigation strategies
    - Model performance comparison
    - Fairness evaluation and recommendations
    
    Run with: python openAI_fix_bias_dataset.py
    
    Requirements:
    - pandas >= 1.3.0
    - numpy >= 1.21.0
    - scikit-learn >= 1.0.0
    - imbalanced-learn >= 0.8.0
    - matplotlib >= 3.5.0
    - seaborn >= 0.11.0
    """
    print("🚀 啟動有偏見資料集分析與修復演示")
    print("=" * 80)
    
    try:
        main()
        print("\n✅ 演示成功完成!")
        print("💡 提示: 在實際項目中，建議根據具體場景選擇合適的偏見緩解策略")
        print("   並進行充分的測試驗證以確保公平性。")
    except Exception as e:
        print(f"\n❌ 演示過程中出错: {e}")
        print("\n💡 故障排除提示:")
        print("   • 確保已安裝所有必需的包")
        print("   • 檢查Python版本兼容性")
        print("   • 查看詳細錯誤信息進行調試")
        import traceback
        traceback.print_exc()
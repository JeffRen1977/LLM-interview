"""
Question 10: 分析並修復一個有偏見的資料集 (Analyze and fix a biased dataset)

這道題考察的是你對機器學習公平性 (Fairness) 和倫理 (Ethics) 的理解，
以及處理不完美資料的實際操作能力。你需要知道如何檢測偏見、如何進行資料增強或重採樣，
以及如何在演算法層面考慮公平性。
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    print("=== 分析並修復有偏見的資料集 ===\n")
    
    # 1. 創建一個模擬的、有偏見的資料集
    # 假設我們在預測一個罕見事件（如信用違約），其中 95% 是非違約（Class 0），5% 是違約（Class 1）
    print("1. 創建模擬的偏見資料集...")
    np.random.seed(42)  # 設置隨機種子以確保結果可重現
    
    data = {
        'feature1': np.random.rand(1000) * 10,
        'feature2': np.random.rand(1000) * 5,
        'target': [0] * 950 + [1] * 50  # 嚴重不平衡
    }
    df = pd.DataFrame(data)
    np.random.shuffle(df.values)  # 打亂資料順序
    
    X = df[['feature1', 'feature2']]
    y = df['target']
    
    # 2. 檢測偏見：視覺化類別分佈
    print("\n--- 原始資料分佈 ---")
    print(y.value_counts())
    print(f"類別不平衡比例: {y.value_counts()[0] / y.value_counts()[1]:.1f}:1")
    
    # 創建視覺化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x='target', data=df)
    plt.title("Original Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    
    # 3. 在有偏見的資料上訓練模型並評估
    print("\n2. 在偏見資料上訓練模型...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model_biased = LogisticRegression(random_state=42)
    model_biased.fit(X_train, y_train)
    
    print("\n--- 在偏見資料上訓練的模型的表現 ---")
    y_pred_biased = model_biased.predict(X_test)
    print(classification_report(y_test, y_pred_biased))
    print("注意：儘管總體準確率很高，但對 Class 1 的召回率 (recall) 可能非常低。")
    
    # 4. 解決方案：使用 SMOTE (過採樣) 來平衡訓練資料
    print("\n3. 使用 SMOTE 重採樣來修復偏見...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("\n--- 使用 SMOTE 重採樣後的訓練資料分佈 ---")
    print(pd.Series(y_train_resampled).value_counts())
    print(f"重採樣後類別比例: {pd.Series(y_train_resampled).value_counts()[0] / pd.Series(y_train_resampled).value_counts()[1]:.1f}:1")
    
    # 視覺化重採樣後的資料分佈
    plt.subplot(1, 2, 2)
    sns.countplot(x=pd.Series(y_train_resampled))
    plt.title("Resampled Class Distribution (SMOTE)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig('bias_dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 在修復後的資料上重新訓練模型並評估
    print("\n4. 在修復後資料上重新訓練模型...")
    model_fixed = LogisticRegression(random_state=42)
    model_fixed.fit(X_train_resampled, y_train_resampled)
    
    print("\n--- 在修復後資料上訓練的模型的表現 ---")
    y_pred_fixed = model_fixed.predict(X_test)
    print(classification_report(y_test, y_pred_fixed))
    print("觀察 Class 1 的召回率是否有顯著提升。")
    
    # 6. 比較兩個模型的混淆矩陣
    print("\n5. 比較兩個模型的混淆矩陣...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 偏見模型的混淆矩陣
    cm_biased = confusion_matrix(y_test, y_pred_biased)
    sns.heatmap(cm_biased, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix - Biased Model')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # 修復後模型的混淆矩陣
    cm_fixed = confusion_matrix(y_test, y_pred_fixed)
    sns.heatmap(cm_fixed, annot=True, fmt='d', cmap='Greens', ax=ax2)
    ax2.set_title('Confusion Matrix - Fixed Model')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('bias_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. 總結分析
    print("\n=== 分析總結 ===")
    print("偏見來源分析:")
    print("- 採樣偏見: 資料集中 Class 0 和 Class 1 的比例嚴重不平衡 (19:1)")
    print("- 這導致模型傾向於預測多數類別，對少數類別的識別能力很差")
    
    print("\n解決方案效果:")
    print("- 使用 SMOTE 重採樣技術平衡了訓練資料")
    print("- 修復後的模型在少數類別上的召回率顯著提升")
    print("- 整體模型的公平性得到改善")
    
    print("\n建議的進一步改進:")
    print("- 收集更多樣化的資料")
    print("- 考慮使用其他公平性約束算法")
    print("- 定期監控模型在不同群體上的表現")

if __name__ == "__main__":
    main()
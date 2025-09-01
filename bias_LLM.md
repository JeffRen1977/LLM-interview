好的，這兩道是針對機器學習和深度學習工程師非常經典的面試題。我將為您提供每道題的詳細解析和可以運行的 Python 程式碼範例。
Question 10: 分析並修復一個有偏見的資料集 (Analyze and fix a biased dataset)
題目:
給你一個存在偏見的資料集，要求你分析偏見的來源並提出解決方案。
(Given a biased dataset, you are asked to analyze the source of the bias and propose a solution.)
考察重點:
這道題考察的是你對機器學習公平性 (Fairness) 和倫理 (Ethics) 的理解，以及處理不完美資料的實際操作能力。你需要知道如何檢測偏見、如何進行資料增強或重採樣，以及如何在演算法層面考慮公平性。
解答與分析 (Answer and Analysis)
1. 分析偏見的來源 (Analyzing the Source of Bias)
資料偏見可能來自多個階段，主要可以分為以下幾類：
 * 採樣偏見 (Sampling Bias): 收集資料的方式導致某些子群體的樣本數量遠多於或少於其他群體。例如，在訓練人臉辨識模型時，如果資料集中絕大多數是白人男性的臉孔，模型在辨識其他族裔或性別時的表現就會很差。
 * 社會偏見 (Societal Bias): 資料反映了現實世界中存在的歷史或社會偏見。例如，在自然語言處理中，歷史文本資料可能會將「醫生」與男性關聯，將「護士」與女性關聯，模型學習後會固化這種刻板印象。
 * 測量偏見 (Measurement Bias): 資料收集的工具或流程存在系統性誤差。例如，在不同地區使用不同品質的攝影機收集圖像資料，可能導致模型對低畫質圖像的處理能力較差。
 * 演算法偏見 (Algorithmic Bias): 模型本身或其優化目標可能加劇或引入偏見。例如，一個以「點擊率」為目標的推薦系統，可能會不斷推薦聳動或極端的內容，因為這類內容更容易吸引點擊。
如何檢測偏見？
 * 探索性資料分析 (EDA): 對資料進行視覺化，檢查不同類別（如性別、種族、地區）的資料分佈是否均衡。
 * 評估指標分解: 將模型的評估指標（如準確率、精確率、召回率）在不同子群體上分別計算。如果模型在某個群體上的表現遠遜於其他群體，就說明存在偏見。例如，計算男性群體和女性群體的模型準確率，觀察是否存在顯著差異。
2. 提出解決方案 (Proposing Solutions)
解決偏見的方案可以從資料、模型和後處理三個層面入手：
 * 資料層面 (Pre-processing):
   * 重採樣 (Resampling): 對樣本數較少的少數群體進行過採樣 (Oversampling)，或對樣本數較多的多數群體進行欠採樣 (Undersampling)。SMOTE (Synthetic Minority Over-sampling Technique) 是一種更高級的過採樣方法。
   * 資料增強 (Data Augmentation): 針對少數群體的資料創造更多樣的訓練樣本。
   * 收集更多樣化的資料: 這是最根本但也是成本最高的解決方案。
 * 演算法層面 (In-processing):
   * 演算法約束: 在模型的損失函數中加入一個懲罰項，用於懲罰模型在不同群體間的不公平性。例如，可以要求模型在不同群體上的預測分佈盡可能相似。
   * 權重調整 (Reweighting): 在模型訓練時，給予少數群體的樣本更高的權重，讓模型更重視它們。
 * 模型輸出層面 (Post-processing):
   * 調整預測閾值: 針對不同群體，使用不同的分類閾值，以達到在各群體間更公平的結果（例如，相似的偽陽性率或偽陰性率）。
示範程式碼 (Example Code)
以下程式碼將展示如何檢測和修復一個常見的偏見問題：類別不平衡 (Class Imbalance)，這是一種典型的採樣偏見。
我們將使用 scikit-learn 和 imbalanced-learn 庫。
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 創建一個模擬的、有偏見的資料集
# 假設我們在預測一個罕見事件（如信用違約），其中 95% 是非違約（Class 0），5% 是違約（Class 1）
data = {
    'feature1': np.random.rand(1000) * 10,
    'feature2': np.random.rand(1000) * 5,
    'target': [0] * 950 + [1] * 50  # 嚴重不平衡
}
df = pd.DataFrame(data)
np.random.shuffle(df.values) # 打亂資料順序

X = df[['feature1', 'feature2']]
y = df['target']

# 2. 檢測偏見：視覺化類別分佈
print("--- 原始資料分佈 ---")
print(y.value_counts())
sns.countplot(x='target', data=df)
plt.title("Original Class Distribution")
plt.show()


# 3. 在有偏見的資料上訓練模型並評估
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model_biased = LogisticRegression()
model_biased.fit(X_train, y_train)

print("\n--- 在偏見資料上訓練的模型的表現 ---")
y_pred_biased = model_biased.predict(X_test)
print(classification_report(y_test, y_pred_biased))
# 注意：儘管總體準確率很高，但對 Class 1 的召回率 (recall) 可能非常低。

# 4. 解決方案：使用 SMOTE (過採樣) 來平衡訓練資料
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\n--- 使用 SMOTE 重採樣後的訓練資料分佈 ---")
print(y_train_resampled.value_counts())


# 5. 在修復後的資料上重新訓練模型並評估
model_fixed = LogisticRegression()
model_fixed.fit(X_train_resampled, y_train_resampled)

print("\n--- 在修復後資料上訓練的模型的表現 ---")
y_pred_fixed = model_fixed.predict(X_test)
print(classification_report(y_test, y_pred_fixed))
# 觀察 Class 1 的召回率是否有顯著提升。

Question 11: 實現一個新穎的注意力變體 (Implement a novel attention variant)
題目:
基於現有的 attention 機制，設計並實現一個新的變種，要求在某些方面有所改進。
(Based on the existing attention mechanism, design and implement a new variant that improves upon it in some way.)
考察重點:
這道題考察的是你的創新能力和對 attention 機制的深度理解。你需要能夠分析現有方法的局限性，提出改進方案，並將其轉化為程式碼進行實現和驗證。
解答與分析 (Answer and Analysis)
1. 分析現有 Attention 機制的局限性
標準的自注意力機制（Scaled Dot-Product Attention）雖然強大，但存在一些局限性：
 * 計算複雜度: 其計算複雜度和記憶體需求都是序列長度 N 的二次方，即 O(N^2)。當處理長序列（如長篇文章、高解析度圖像）時，計算成本會變得非常高。
 * 全域依賴的必要性: 對於某些任務，並非所有的 token 都需要與其他所有 token 計算關聯性。例如，在語言模型中，一個詞的語義可能主要由其附近的詞決定。全域計算有時會引入不必要的噪聲。
2. 設計一個新的 Attention 變體
針對 O(N^2) 的複雜度問題，我們可以設計一個滑動窗口注意力 (Sliding Window Attention) 或 局部注意力 (Local Attention)。
核心思想:
對於序列中的每一個 token，我們不再讓它與序列中的所有其他 token 計算注意力分數，而是只與其左右一個固定大小的窗口（window size, w）內的 token 進行計算。
改進之處:
 * 降低複雜度: 這種方法將計算複雜度從 O(N^2) 降低到 O(N \\cdot w)，其中 w 是一個遠小於 N 的常數。
 * 專注局部信息: 強調了局部上下文的重要性，對於很多任務來說這是一種有效的歸納偏置 (inductive bias)。
這個想法並非全新的（例如 Longformer 和 BigBird 等模型中已有應用），但在面試中，能夠清晰地闡述其動機、設計並實現它，足以展現你對注意力機制的深刻理解。
示範程式碼 (Example Code)
以下程式碼將使用 PyTorch 實現一個 SlidingWindowAttention 模組。
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SlidingWindowAttention(nn.Module):
    """
    實現一個滑動窗口注意力機制 (Sliding Window Attention)。
    """
    def __init__(self, embed_dim, num_heads, window_size):
        """
        初始化函數。
        Args:
            embed_dim (int): 輸入的嵌入維度。
            num_heads (int): 注意力頭的數量。
            window_size (int): 注意力窗口的大小 (單邊)。例如，window_size=2 意味著關注左右各2個 token。
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        # 線性層用於生成 Q, K, V
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        """
        前向傳播。
        Args:
            x (Tensor): 輸入張量，形狀為 (batch_size, seq_len, embed_dim)。
            mask (Tensor, optional): 可選的 padding mask。
        Returns:
            Tensor: 注意力輸出的張量，形狀與輸入 x 相同。
        """
        B, N, C = x.shape  # B=Batch Size, N=Sequence Length, C=Embedding Dimension

        # 1. 生成 Q, K, V
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: (B, num_heads, N, head_dim)

        # 2. 計算注意力分數
        # q @ k.transpose(-2, -1) 的形狀是 (B, num_heads, N, N)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3. 創建並應用滑動窗口 Mask (核心改進點)
        # 我們希望每個 token 只關注其窗口內的 token
        window_mask = torch.ones_like(attn_scores, dtype=torch.bool)
        for i in range(N):
            start = max(0, i - self.window_size)
            end = min(N, i + self.window_size + 1)
            window_mask[:, :, i, start:end] = False # False 的地方是我們希望保留的

        # 將窗口外的注意力分數設置為一個非常小的值
        attn_scores.masked_fill_(window_mask, float('-inf'))
        
        # 如果有 padding mask，也要應用
        if mask is not None:
             attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # 4. Softmax 和加權求和
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # 處理因 mask 導致的 NaN (當一行全是 -inf 時)
        attn_probs = torch.nan_to_num(attn_probs)

        output = attn_probs @ v  # Shape: (B, num_heads, N, head_dim)

        # 5. Reshape and Final Projection
        output = output.transpose(1, 2).reshape(B, N, self.embed_dim)
        output = self.out_proj(output)

        return output

# --- 使用範例 ---
if __name__ == '__main__':
    # 參數設定
    batch_size = 4
    seq_len = 64 # 較長的序列，更能體現優勢
    embed_dim = 128
    num_heads = 8
    window_size = 4 # 每個 token 關注左右各 4 個 token

    # 創建模型和輸入
    attention_variant = SlidingWindowAttention(embed_dim, num_heads, window_size)
    input_tensor = torch.randn(batch_size, seq_len, embed_dim)

    # 運行模型
    output_tensor = attention_variant(input_tensor)

    # 檢查輸出形狀
    print("輸入張量形狀:", input_tensor.shape)
    print("輸出張量形狀:", output_tensor.shape)
    assert input_tensor.shape == output_tensor.shape
    print("\nSliding Window Attention 模組運行成功！")


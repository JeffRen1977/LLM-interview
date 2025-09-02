實現一個新穎的注意力變體 (Implement a novel attention variant)
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


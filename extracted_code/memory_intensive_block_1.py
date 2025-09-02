class MemoryIntensiveBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4 * hidden_dim, hidden_dim)
    
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
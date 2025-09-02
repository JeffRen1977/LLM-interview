graph TD
    A[Input Embedding] --> C{Add}
    B[Positional Encoding] --> C
    C --> D[Multi-Head Attention]
    C --> D
    D --> E{Add & Norm}
    D --> E
    E --> F[Feed-Forward Network]
    E --> F
    F --> G{Add & Norm}
    F --> G
    G --> H[Output]
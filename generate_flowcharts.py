import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs('figures', exist_ok=True)

# ==============================================================================
# 1. FLOWCHART METODOLOGI PENELITIAN (HIGH-RES PUBLICATION READY, DPI 300)
# ==============================================================================
def create_flowchart_metodologi():
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('#ffffff')
    ax.axis('off')

    # Helper function to draw rounded box with shadow
    def draw_box(x, y, w, h, title, subtitle="", bg_color="#ffffff", border_color="#334155", title_color="#0f172a", text_color="#334155"):
        # Shadow
        shadow = patches.FancyBboxPatch((x + 0.005, y - 0.005), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
                                        facecolor="#000000", edgecolor="none", alpha=0.08, zorder=1)
        ax.add_patch(shadow)
        # Main box
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
                                     facecolor=bg_color, edgecolor=border_color, linewidth=2, zorder=2)
        ax.add_patch(box)
        # Text
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=11, fontweight='bold', color=title_color, zorder=3)
            ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=9, color=text_color, zorder=3)
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=11, fontweight='bold', color=title_color, zorder=3)

    # Helper function for arrow
    def draw_arrow(x1, y1, x2, y2, color="#475569", width=2.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=width, mutation_scale=16),
                    zorder=4)

    # 1. Header Title
    ax.text(0.5, 0.96, "DIAGRAM ALIR METODOLOGI PENELITIAN", ha='center', va='center', fontsize=16, fontweight='bold', color='#0f172a')
    ax.text(0.5, 0.925, "Studi Komparasi 3 Paradigma Arsitektur untuk Klasifikasi Berita Hoaks & Fakta Bahasa Indonesia", ha='center', va='center', fontsize=11, color='#475569')

    # 2. Stage 1: Dataset
    draw_box(0.12, 0.78, 0.76, 0.10,
             "1. Integrasi Multi-Dataset Berita Indonesia (Total: 41.258 Artikel)",
             "Archive Politik (31.332 data) + Live Scraped Multi-Topik (10.118 data baru dari 8 Sumber Media Terpercaya)",
             bg_color="#e0f2fe", border_color="#0284c7", title_color="#0369a1")

    # Arrow 1 -> 2
    draw_arrow(0.5, 0.78, 0.5, 0.70)

    # 3. Stage 2: Preprocessing
    draw_box(0.18, 0.60, 0.64, 0.10,
             "2. Pembersihan & Standardisasi Teks (Text Preprocessing)",
             "Penyelamatan Teks Utuh (extract_best_text) • Case Folding • Pembersihan Regex • Normalisasi Whitespace",
             bg_color="#f1f5f9", border_color="#64748b", title_color="#1e293b")

    # Arrow 2 -> 3
    draw_arrow(0.5, 0.60, 0.5, 0.52)

    # 4. Stage 3: Stratified Splitting
    draw_box(0.15, 0.42, 0.70, 0.10,
             "3. Pembagian Data Uji Terstratifikasi (Stratified Splitting 80 : 10 : 10)",
             "Training Set: 33.006 data (80%)  |  Validation Set: 4.126 data (10%)  |  Test Set: 4.126 data (10% Identik)",
             bg_color="#fef3c7", border_color="#d97706", title_color="#b45309")

    # Arrows 3 -> 3 Models
    draw_arrow(0.28, 0.42, 0.18, 0.33)
    draw_arrow(0.50, 0.42, 0.50, 0.33)
    draw_arrow(0.72, 0.42, 0.82, 0.33)

    # 5. Stage 4: Three Architectures
    # Model 1
    draw_box(0.04, 0.21, 0.28, 0.12,
             "Paradigma 1: ML Konvensional",
             "LinearSVC + TF-IDF (10.000 Fitur)\nUnigram & Bigram + Sublinear TF\nPlatt Scaling (CalibratedClassifierCV)",
             bg_color="#ecfdf5", border_color="#059669", title_color="#047857")

    # Model 2
    draw_box(0.36, 0.21, 0.28, 0.12,
             "Paradigma 2: Deep Learning",
             "TextCNN (Yoon Kim 2014)\nDense Embedding (128-d) + Parallel Conv1D\nFilter (3, 4, 5 kata) + Max-over-Time",
             bg_color="#eff6ff", border_color="#2563eb", title_color="#1d4ed8")

    # Model 3
    draw_box(0.68, 0.21, 0.28, 0.12,
             "Paradigma 3: Transformer",
             "IndoBERT Base Phase 1\nWordPiece (30.521) + 12 Multi-Head Attention\nBidirectional Self-Attention + [CLS] Token",
             bg_color="#faf5ff", border_color="#7c3aed", title_color="#6d28d9")

    # Arrows 3 Models -> Stage 5 Evaluation
    draw_arrow(0.18, 0.21, 0.38, 0.11)
    draw_arrow(0.50, 0.21, 0.50, 0.11)
    draw_arrow(0.82, 0.21, 0.62, 0.11)

    # 6. Stage 5: Evaluation & XAI
    draw_box(0.12, 0.03, 0.76, 0.08,
             "5. Evaluasi Uji Identik, Komparasi Kinerja Jurnal & Explainable AI (SHAP)",
             "Metrik: Accuracy, Macro F1, Weighted F1, Precision, Recall, ROC-AUC  |  Interpretasi Atribusi Fitur SHAP XAI",
             bg_color="#fdf2f8", border_color="#db2777", title_color="#be185d")

    plt.tight_layout()
    plt.savefig('figures/flowchart_metodologi.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("figures/flowchart_metodologi.png created successfully!")

# ==============================================================================
# 2. DIAGRAM MEKANISME FITUR 3 ARSITEKTUR (DPI 300)
# ==============================================================================
def create_feature_mechanism_diagram():
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    for ax in axes:
        ax.set_facecolor('#f8fafc')
        ax.axis('off')

    # Draw helper for subplot
    def draw_sub_box(ax, x, y, w, h, title, subtitle="", bg_color="#ffffff", border_color="#334155", title_color="#0f172a"):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
                                     facecolor=bg_color, edgecolor=border_color, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=title_color, zorder=3)
            ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=8, color="#334155", zorder=3)
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=title_color, zorder=3)

    def draw_sub_arrow(ax, x1, y1, x2, y2, color="#64748b"):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=12),
                    zorder=4)

    # ------------------ SUBPLOT 1: SVM + TF-IDF ------------------
    ax1 = axes[0]
    ax1.text(0.02, 0.85, "1. SVM + TF-IDF (Lexical / Statistical Feature Mechanism)", fontsize=12, fontweight='bold', color="#047857")
    draw_sub_box(ax1, 0.02, 0.20, 0.16, 0.50, "Teks Paragraf", "Rata-rata 234 kata", bg_color="#f1f5f9", border_color="#94a3b8")
    draw_sub_arrow(ax1, 0.18, 0.45, 0.22, 0.45)
    draw_sub_box(ax1, 0.22, 0.20, 0.24, 0.50, "TF-IDF Vectorizer", "Unigram (1) + Bigram (2)\nSublinear TF: 1 + log(tf)\nSmooth IDF: log((1+N)/(1+df))+1", bg_color="#ecfdf5", border_color="#059669", title_color="#047857")
    draw_sub_arrow(ax1, 0.46, 0.45, 0.50, 0.45)
    draw_sub_box(ax1, 0.50, 0.20, 0.20, 0.50, "Sparse Feature Vector", "x ∈ ℝ¹⁰⁰⁰⁰ (L2 Normalized)\n||x||₂ = 1", bg_color="#ecfdf5", border_color="#059669", title_color="#047857")
    draw_sub_arrow(ax1, 0.70, 0.45, 0.74, 0.45)
    draw_sub_box(ax1, 0.74, 0.20, 0.24, 0.50, "LinearSVC + Platt Scaling", "f(x) = wᵀx + b\nP(y=1|x) = σ(A·f(x) + B)", bg_color="#d1fae5", border_color="#047857", title_color="#065f46")

    # ------------------ SUBPLOT 2: TextCNN ------------------
    ax2 = axes[1]
    ax2.text(0.02, 0.85, "2. TextCNN (Multi-Filter N-Gram Convolution Mechanism)", fontsize=12, fontweight='bold', color="#1d4ed8")
    draw_sub_box(ax2, 0.02, 0.20, 0.16, 0.50, "Tokenisasi Kata", "Vocab: 30.000 kata\nLength L = 128 token", bg_color="#f1f5f9", border_color="#94a3b8")
    draw_sub_arrow(ax2, 0.18, 0.45, 0.22, 0.45)
    draw_sub_box(ax2, 0.22, 0.20, 0.18, 0.50, "Dense Embedding", "nn.Embedding(30k, 128)\nMatriks X ∈ ℝ¹²⁸ˣ¹²⁸", bg_color="#eff6ff", border_color="#2563eb", title_color="#1d4ed8")
    draw_sub_arrow(ax2, 0.40, 0.45, 0.44, 0.45)
    draw_sub_box(ax2, 0.44, 0.20, 0.26, 0.50, "Parallel 1D Convolutions", "Filter k=3 (trigram) × 100\nFilter k=4 (4-gram) × 100\nFilter k=5 (5-gram) × 100", bg_color="#eff6ff", border_color="#2563eb", title_color="#1d4ed8")
    draw_sub_arrow(ax2, 0.70, 0.45, 0.74, 0.45)
    draw_sub_box(ax2, 0.74, 0.20, 0.24, 0.50, "Max-Pooling & Dense FC", "Global Max-over-Time: z ∈ ℝ³⁰⁰\nDropout (0.5) -> Linear FC -> Softmax", bg_color="#dbeafe", border_color="#1d4ed8", title_color="#1e40af")

    # ------------------ SUBPLOT 3: IndoBERT ------------------
    ax3 = axes[2]
    ax3.text(0.02, 0.85, "3. IndoBERT (Bidirectional Multi-Head Self-Attention Mechanism)", fontsize=12, fontweight='bold', color="#6d28d9")
    draw_sub_box(ax3, 0.02, 0.20, 0.16, 0.50, "WordPiece Sub-word", "30.521 Sub-words\nBebas masalah OOV", bg_color="#f1f5f9", border_color="#94a3b8")
    draw_sub_arrow(ax3, 0.18, 0.45, 0.22, 0.45)
    draw_sub_box(ax3, 0.22, 0.20, 0.22, 0.50, "Input Embeddings", "Token + Position + Segment\nDimensi d_model = 768", bg_color="#faf5ff", border_color="#7c3aed", title_color="#6d28d9")
    draw_sub_arrow(ax3, 0.44, 0.45, 0.48, 0.45)
    draw_sub_box(ax3, 0.48, 0.20, 0.26, 0.50, "12 Layers Transformer", "12 Attention Heads per layer\nAttention(Q,K,V) = softmax(QKᵀ/√d)V\nResidual + LayerNorm + FFN", bg_color="#faf5ff", border_color="#7c3aed", title_color="#6d28d9")
    draw_sub_arrow(ax3, 0.74, 0.45, 0.78, 0.45)
    draw_sub_box(ax3, 0.78, 0.20, 0.20, 0.50, "[CLS] Classifier Head", "Pooled Vector h_[CLS] ∈ ℝ⁷⁶⁸\nLinear Projection -> Softmax", bg_color="#ede9fe", border_color="#6d28d9", title_color="#5b21b6")

    plt.tight_layout()
    plt.savefig('figures/diagram_fitur_3_arsitektur.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("figures/diagram_fitur_3_arsitektur.png created successfully!")

if __name__ == '__main__':
    create_flowchart_metodologi()
    create_feature_mechanism_diagram()


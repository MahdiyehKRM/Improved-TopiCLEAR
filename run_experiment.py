import numpy as np
from sklearn.datasets import fetch_20newsgroups
from experiments.methods.topiclear_runner import run_topiclear
from experiments.evaluation import evaluate_clustering
import time

def main():
    print("--- 1. Loading Dataset (20 Newsgroups subset) ---")
    # انتخاب ۴ دسته با تم‌های متفاوت
    categories = [
        'sci.space',
        'rec.sport.baseball',
        'talk.politics.mideast',
        'comp.graphics'
    ]
    
    # لود داده‌ها (فقط متن‌های تمیز بدون هدر و فوتر)
    dataset = fetch_20newsgroups(
        subset='test',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    texts = dataset.data
    true_labels = np.array(dataset.target)
    
    # حذف متن‌های خالی یا بسیار کوتاه
    filtered_data = [(t, l) for t, l in zip(texts, true_labels) if len(t.strip()) > 30]
    texts = [item[0] for item in filtered_data]
    true_labels = np.array([item[1] for item in filtered_data])
    
    n_topics = len(categories)
    print(f"Total documents to evaluate: {len(texts)} across {n_topics} categories.")
    
    print("\n--- 2. Running TopiCLEAR (Baseline Model) ---")
    start_time = time.time()
    
    # اجرای مدل با تنظیمات پیش‌فرض
    model, pred_labels = run_topiclear(
        X=texts,
        n_topics=n_topics,
        input_type="text",
        random_state=42,
        embedding_show_progress_bar=True
    )
    
    elapsed = time.time() - start_time
    print(f"Clustering finished in {elapsed:.2f} seconds.")
    
    print("\n--- 3. Evaluation Results ---")
    results = evaluate_clustering(true_labels, pred_labels)
    print(f"ARI (Adjusted Rand Index): {results['ari']:.4f}")
    print(f"AMI (Adjusted Mutual Info): {results['ami']:.4f}")

if __name__ == "__main__":
    main()

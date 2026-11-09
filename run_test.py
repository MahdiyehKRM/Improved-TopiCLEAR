from topiclear import TopiCLEAR

# ۱. چند متن نمونه از سه موضوع مختلف (فناوری، پزشکی/سلامت، و ورزش)
sample_docs = [
    "Artificial intelligence and deep learning models are advancing rapidly.",
    "Machine learning algorithms require clean training datasets and GPUs.",
    "Neural networks and transformers process human natural language.",
    
    "Doctors recommend cardiovascular exercise and a balanced diet for health.",
    "The clinical trial evaluated patient recovery and medical symptoms.",
    "Regular sleep and nutrition improve the human immune system.",
    
    "The football team won the championship match after scoring two goals.",
    "Basketball players run intense drills to prepare for the tournament.",
    "The athletic competition concluded with a gold medal for the runner."
]

print("Initializing TopiCLEAR model...")
# ساخت مدل با ۳ کلاستر و ورودی متنی
model = TopiCLEAR(n_clusters=3, input_type="text", random_state=42)

print("Fitting model and extracting topics (this downloads the MiniLM model once if not cached)...")
model.fit(sample_docs)

# پیش‌بینی لیبل هر متن
labels = model.predict(sample_docs)

print("\n" + "="*40)
print("RESULTS / CLUSTERING OUTPUT:")
print("="*40)
for doc, label in zip(sample_docs, labels):
    print(f"[{label}] -> {doc}")

print("\nTop words per cluster/topic (Interpretation):")
if hasattr(model, "get_topics"):
    topics = model.get_topics()
    for topic_id, words in topics.items():
        print(f"Topic {topic_id}: {words}")
else:
    print(f"Clusters successfully assigned: {set(labels)}")

print("\nDone! TopiCLEAR is fully operational.")

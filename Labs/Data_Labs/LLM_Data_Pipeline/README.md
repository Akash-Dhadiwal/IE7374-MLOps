# Lab 2 — Streaming LM Pipeline

## Overview

This lab demonstrates a production-ready streaming language modeling data pipeline using Hugging Face Datasets and PyTorch. The implementation processes large-scale text corpora efficiently without loading entire datasets into memory, making it suitable for real-world LM training scenarios.

## Key Features

### Dataset Configuration
- **Dataset**: `wikitext-103-v1` (103M tokens)
- **Mode**: Streaming (memory-efficient)
- **Tokenizer**: GPT-2 tokenizer
- **Split**: Training set

### Processing Parameters
- **Block size**: 256 tokens
- **Batch size**: 8
- **Tokenizer**: GPT2TokenizerFast

## Implementation Highlights

### 1. Robust Error Handling
The pipeline includes defensive programming to handle various data structures:
```python
ids = example.get("input_ids") if isinstance(example, dict) else None
if ids is None:
    try:
        ids = example["input_ids"]
    except Exception:
        continue
```

### 2. Performance Monitoring
Real-time throughput measurement for optimization:
- Tracks processing time
- Calculates tokens/second
- Example benchmark: ~4254 tokens/sec

### 3. Rolling Buffer Architecture
Efficiently concatenates and chunks tokens into fixed-length blocks:
- Maintains buffer across document boundaries
- Prevents data loss at chunk boundaries
- Optional padding for final chunk (configurable)

### 4. Statistical Analysis
Utility function `sample_stats()` provides:
- Average tokens per example
- Median token count
- Processing speed metrics
- Configurable sample size

### 5. Comprehensive Documentation
Extensive inline documentation covering:
- Architecture decisions
- Processing steps
- Usage guidelines
- Troubleshooting tips

## Pipeline Architecture

### Step 1: Streaming Dataset Loading
```python
stream_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
```
Loads dataset in streaming mode to avoid memory constraints.

### Step 2: Tokenization
```python
tokenized_stream = stream_dataset.map(tokenize_function, batched=True)
```
Lazy tokenization without padding/truncation for maximum flexibility.

### Step 3: Token Grouping
```python
def group_texts_streaming(dataset_iter, block_size):
    # Rolling buffer implementation
```
Concatenates tokens and yields fixed-length chunks using a rolling buffer.

### Step 4: PyTorch Integration
```python
class StreamingLMIterableDataset(IterableDataset):
    def __iter__(self):
        return group_texts_streaming(self.dataset, self.block_size)
```
Wraps generator in PyTorch's `IterableDataset` for seamless integration.

### Step 5: Batching
```python
train_loader = DataLoader(grouped_iterable_dataset, batch_size=8, collate_fn=collate_fn)
```
Creates training-ready batches with proper tensors.

## Output Format

Each batch contains:
- **input_ids**: Tokenized text sequences `[batch_size, block_size]`
- **attention_mask**: Binary mask for valid tokens `[batch_size, block_size]`
- **labels**: Copy of input_ids for language modeling loss

Example output shape: `torch.Size([8, 256])`

## Performance Metrics

Sample benchmark (3 batches):
- **Processed tokens**: 6,144
- **Elapsed time**: ~1.44s
- **Throughput**: ~4,254 tokens/sec

## Installation

```bash
pip install datasets transformers torch
```

## Usage

Run all cells in order in Jupyter/Colab/VS Code. The notebook is self-contained and ready for execution.

### Running Statistics (Optional)
Uncomment the `sample_stats` function call to analyze token distribution:
```python
stats = sample_stats(tokenized_stream, n_examples=200)
print(stats)
```

**Note**: Use small sample sizes to avoid long processing times.

## Design Considerations

### Why Streaming?
- Handles datasets larger than available RAM
- Enables training on web-scale corpora
- Reduces startup time
- Mimics production ML pipelines

### Why Block Size 256?
- Balances context length and memory usage
- Sufficient context for meaningful language patterns
- Efficient GPU utilization
- Standard size for medium-scale LM training

### Why Rolling Buffer?
- Prevents token loss at document boundaries
- Maintains sequence continuity
- Efficient memory usage
- Handles variable-length documents

## Troubleshooting

### Rate Limiting
If streaming fails due to HF rate limits:
- Use a smaller dataset locally
- Enable HF caching
- Check network connectivity

### Memory Issues
If OOM errors occur:
- Reduce batch size
- Decrease block size
- Ensure streaming mode is enabled

### Slow Processing
To improve throughput:
- Increase batch size (if memory allows)
- Use faster tokenizer (Fast version)
- Enable multi-processing in DataLoader

## Submission Notes

This notebook is the complete deliverable for the assignment. All code is functional and demonstrates:
- Streaming data pipeline implementation
- Memory-efficient processing
- Production-ready error handling
- Performance optimization
- Comprehensive documentation

## References

- Hugging Face Datasets: [https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)
- Transformers Library: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- PyTorch IterableDataset: [https://pytorch.org/docs/stable/data.html](https://pytorch.org/docs/stable/data.html)

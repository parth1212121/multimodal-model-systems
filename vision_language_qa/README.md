# Vision-Language QA

Vision-language question answering built around a frozen visual encoder, a learned projection module, and an LLM adapter.

## Example

```bash
python inference.py \
  --model_dir /path/to/model_dir \
  --data_path /path/to/questions.json \
  --output_file outputs/qa_predictions.json
```

The input should be a CLEVR-style question JSON file with image paths and question records.


# MAIDS
inference：

step 1.
pip install --upgrade metagpt

step 2.
metagpt --init-config

step 3. 
modify your own config.yaml as follow with your openai_key

llm:
  api_type: "openai"
  api_version: "2024-02-15-preview"
  base_url: "your_url"
  api_key: 'your_key'
  model: "openai"

step 4. cd ./MetaGPT-main/examples/clinic

step 5. python MAgent.py

Model_download_url:xxx
Dataset_download_url: xxx

## Citation

To cite [MAIDS](XXX) in publications, please use the following BibTeX entries.

```bibtex

```


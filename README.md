# DP Clinical Coding

This repository contains code and experiments from the MSc thesis  
*Comparing Privacy-Preserving Pipelines for Clinical ICD-9 Classification* (Imperial College London, 2025).  

It implements and evaluates four training pipelines on the MIMIC-III dataset:

- **LoRA-No-DP**: Baseline 1B LoRA fine-tuning without DP  
- **DP-Small**: Direct DP-SGD training of a 1B model  
- **DP-Synthetic**: DP-trained 3B generator → synthetic data → 1B classifier  
- **DP-Distil**: DP-trained 3B generator + teacher → synthetic data + logits → 1B student  

All pipelines produce 1B-parameter classifiers on the top-50 ICD-9 codes.  
All final classifiers, except the LoRA-No-DP baseline, satisfy a formal $(\varepsilon,\delta)$-DP guarantee.

---

## Requirements

- Python 3.10+
- CUDA 12.x, PyTorch ≥ 2.1
- `transformers`, `datasets`, `peft`
- `opacus`, `scikit-learn`, `tqdm`
- `numpy`, `pandas`

Install everything with:

```bash
pip install -r requirements.txt
````

---

## Running the Pipelines

Run scripts in order:

1. **Prepare data**

   ```bash
   python 01_prepare_mimic_data.py
   ```

   → writes `data/real/real_{train,val,test}_codes_notes.pt`

2. **Train generators & make synthetic data**

   ```bash
   bash 02_run_train_gen_synthetic.sh
   ```

3. **Train classifiers**

   ```bash
   bash 03_run_train_classifiers.sh
   ```

4. **Evaluate classifiers**

   ```bash
   bash 04_run_eval_classifiers.sh
   ```

---

## GPUs

Experiments were run on NVIDIA **RTX 6000 Ada (48 GB VRAM)** GPUs.
Select which GPU(s) to use by setting the `GPUS` variable:

```bash
# run on GPU 1
GPUS="1" bash 03_run_train_classifiers.sh
```

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)** –
see the [LICENSE](https://github.com/mathieu-dufour/dp-clinical-coding/blob/main/LICENSE) file for details.

"""
eScreen GUI — Streamlit interface for prediction, interpretability, and training.
Usage:
    python -m escreen.serve --hostname 0.0.0.0 --port 8501
"""
import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def one_hot_encode(seq: str, max_len: int = 512) -> np.ndarray:
    arr = np.zeros((max_len, 4), dtype=np.uint8)
    for j, b in enumerate(seq.upper()[:max_len]):
        if b in BASE_MAP:
            arr[j, BASE_MAP[b]] = 1
    return arr


@st.cache_resource
def init_model():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_model = 256

    from escreen import eSCREEN_backbone, eScreen_vX
    from escreen.motif import load_pwm_from_meme_c
    from escreen.CellEmb import Geneformer

    meme_file = str(DATA_DIR / "consensus_pwms.meme")
    motifs_f, motifs_r, _, _ = load_pwm_from_meme_c(meme_file, max_length=35)
    kernel_fwd = torch.tensor(motifs_f, dtype=torch.float)
    kernel_rev = torch.tensor(motifs_r, dtype=torch.float)

    with open(DATA_DIR / "DepMap_CellLine_embedding.pkl", "rb") as f:
        GF_predefined_emb, GFcelltype2token = pickle.load(f)
    GF_predefined_emb = torch.stack(GF_predefined_emb)
    token2GFcelltype = {v: k for k, v in GFcelltype2token.items()}

    gf_dicts = DATA_DIR / "geneformer" / "gene_dictionaries_30m"
    GF = Geneformer(
        TOKEN_DICTIONARY_FILE=str(gf_dicts / "token_dictionary_gc30M.pkl"),
        GENE_MEDIAN_FILE=str(gf_dicts / "gene_median_dictionary_gc30M.pkl"),
        ENSEMBL_DICTIONARY_FILE=str(gf_dicts / "gene_name_id_dict_gc30M.pkl"),
        ENSEMBL_MAPPING_FILE=str(gf_dicts / "ensembl_mapping_dict_gc30M.pkl"),
        MODEL_PATH=str(DATA_DIR / "geneformer" / "Geneformer-V1-10M"),
        PREDIFINED_EMB=GF_predefined_emb,
        model_version="V1",
        device=DEVICE,
    )

    backbone = eSCREEN_backbone(
        filter_type='pwm', kernel_fwd=kernel_fwd, kernel_rev=kernel_rev,
        d_model=d_model, num_filters=256,
        use_flash_attn=False, proj_groups=1,
        seq_length=512, celltype_num=64, lr=1e-5, device=DEVICE,
    )

    model = eScreen_vX(
        backbone=backbone.to(DEVICE),
        cell_emb=GF,
        d_model=d_model,
        cell_lora_rank=64, MoE_K=64,
        freeze_backbone=False, freeze_cellemb=False,
        freeze_celllora=False, freeze_header=False,
    ).to(DEVICE)

    ckpt = DATA_DIR / "best_model.pt"
    if ckpt.exists():
        model.load_state_dict(
            torch.load(str(ckpt), map_location=DEVICE, weights_only=False),
            strict=False,
        )

    model.eval()
    return model, DEVICE, token2GFcelltype, GFcelltype2token


def df_to_data(df: pd.DataFrame, ct2idx: dict, seq_len: int = 512):
    seqs = []
    cts = []
    for _, row in df.iterrows():
        seqs.append(one_hot_encode(str(row['sequence']), seq_len))
        ct_name = str(row['celltype'])
        if ct_name not in ct2idx:
            ct2idx[ct_name] = len(ct2idx)
        cts.append(ct2idx[ct_name])
    return {
        'sequence': np.stack(seqs),
        'cell_type': np.array(cts, dtype=np.int32),
        'length': len(df),
    }, ct2idx


def run_attribute(model, seqs, ct_indices, device, batch_size=8):
    from captum.attr import IntegratedGradients
    model.eval()
    ig = IntegratedGradients(model.forward)
    all_attr = []
    n = len(seqs)
    for i in range(0, n, batch_size):
        bx = torch.tensor(seqs[i:i + batch_size], dtype=torch.float, device=device).requires_grad_()
        bc = torch.tensor(ct_indices[i:i + batch_size], dtype=torch.int, device=device)
        bl = torch.zeros_like(bx)
        attr, _ = ig.attribute(
            bx, baselines=bl, additional_forward_args=bc,
            return_convergence_delta=True, method="gausslegendre", n_steps=30,
        )
        score = (attr.detach().cpu().numpy() * bx.detach().cpu().numpy()).sum(axis=-1)
        all_attr.append(score)
    return np.concatenate(all_attr, axis=0)


def main():
    parser = argparse.ArgumentParser(description="eScreen GUI")
    parser.add_argument("--hostname", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    env = os.environ.copy()
    env["_ESCREEN_SERVE"] = "1"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", __file__,
        "--server.address", args.hostname,
        "--server.port", str(args.port),
        "--server.headless", "true",
    ], env=env)


if os.environ.get("_ESCREEN_SERVE") == "1":
    st.set_page_config(page_title="eScreen", layout="wide")
    st.title("eScreen — functional CRE screening")

    model, DEVICE, token2celltype, celltype2token = init_model()
    known_celltypes = sorted(celltype2token.keys())

    tab_pred, tab_attr, tab_train = st.tabs(["Predict", "Attribution", "Train"])

    with tab_pred:
        st.subheader("Predict regulatory activity")
        pred_file = st.file_uploader(
            "Upload BED file (TSV, columns: sequence, celltype)",
            type=['bed', 'tsv', 'txt'], key='pred',
        )
        pred_batch = st.number_input("Batch size", 8, 512, 128, key='pred_bs')
        pred_btn = st.button("Predict", key='pred_btn')

        if pred_btn and pred_file is not None:
            df = pd.read_csv(pred_file, sep='\t', header=None).iloc[:, :2]
            df.columns = ['sequence', 'celltype']
            ct2idx = {}
            data, ct2idx = df_to_data(df, ct2idx)
            preds, _ = model.predict(data, batch_size=int(pred_batch), device=DEVICE, verbose=False)
            out = df.copy()
            out['prediction'] = preds
            st.dataframe(out, use_container_width=True)
            csv = out.to_csv(index=False, sep='\t')
            st.download_button("Download results", csv, "predictions.tsv")

    with tab_attr:
        st.subheader("Attribution (Integrated Gradients)")
        col1, col2 = st.columns([3, 1])
        with col1:
            attr_seq = st.text_area(
                "DNA sequence (max 512 bp)", height=120,
                placeholder="e.g. ACGTACGT...",
            )
        with col2:
            attr_ct_input = st.text_input(
                "Cell type", placeholder="Type cell line name...",
                key='attr_ct',
            )
            attr_ct = None
            if attr_ct_input:
                matches = [c for c in known_celltypes
                           if attr_ct_input.lower() in c.lower()]
                if not matches:
                    st.error(f"Unknown cell type: {attr_ct_input}")
                elif len(matches) > 1:
                    sel = st.selectbox("Suggestions", [""] + matches,
                                       format_func=lambda x: x or "-- select --",
                                       key='attr_ct_suggest')
                    attr_ct = sel if sel else None
                else:
                    attr_ct = matches[0]
                    st.caption(f"→ {attr_ct}")

        attr_btn = st.button("Run attribution")

        if attr_btn and attr_seq.strip() and attr_ct is not None:
            seq = attr_seq.strip().upper()[:512]
            oh = one_hot_encode(seq, 512).reshape(1, 512, 4)
            ct = np.array([celltype2token[attr_ct]], dtype=np.int32)
            attr = run_attribute(model, oh, ct, DEVICE, batch_size=1)[0]
            positions = list(range(len(seq)))
            df_attr = pd.DataFrame({'position': positions, 'importance': attr[:len(seq)]})
            st.line_chart(df_attr.set_index('position'))
            st.caption("Per-nucleotide importance (sum across 4 channel × gradient)")

    with tab_train:
        st.subheader("Fine-tune eScreen")
        train_file = st.file_uploader(
            "Upload BED file (TSV, columns: sequence, celltype, label, [score])",
            type=['bed', 'tsv', 'txt'], key='train',
        )

        with st.expander("Training hyperparameters"):
            c1, c2 = st.columns(2)
            with c1:
                train_epochs = st.number_input("Epochs", 1, 200, 20)
                train_lr = st.number_input("Learning rate", 1e-6, 1e-2, 3e-4, format='%.6f')
                train_bs = st.number_input("Batch size", 16, 1024, 256)
                train_wd = st.number_input("Weight decay", 0.0, 0.1, 0.01, format='%.4f')
            with c2:
                task_type = st.selectbox("Task", ["reg", "cls"])
                use_boost = st.checkbox("Boost resampling", True)
                boost_t = st.slider("Boost temperature", 0.1, 2.0, 0.6, 0.05)
                earlystop = st.number_input("Early stop patience", 1, 50, 10)
                aux_bce = st.number_input("Aux BCE lambda", 0.0, 2.0, 0.6, 0.1)
                pos_weight = st.number_input("Pos weight", 0.0, 100.0, 30.0, 1.0)

        train_btn = st.button("Start training")

        if train_btn and train_file is not None:
            df = pd.read_csv(train_file, sep='\t', header=None)
            ncol = df.shape[1]
            if ncol < 3:
                st.error("Need at least 3 columns: sequence, celltype, label")
                st.stop()
            names = ['sequence', 'celltype', 'label']
            if ncol >= 4:
                names.append('score')
            df.columns = names[:ncol]

            has_score = 'score' in df.columns
            ct2idx = {}
            data, ct2idx = df_to_data(df, ct2idx)
            data['y'] = df['score'].values.astype(np.float32) if has_score else df['label'].values.astype(np.float32)
            has_binary = df['label'].dropna().nunique() <= 2
            if has_binary:
                data['label'] = df['label'].values.astype(np.float32)

            opt = torch.optim.AdamW(model.parameters(), lr=float(train_lr), weight_decay=float(train_wd))

            model.train()
            model.fit(
                train_data=data,
                batch_size=int(train_bs),
                epochs=int(train_epochs),
                optimizer=opt,
                earlystop=int(earlystop),
                use_boost=use_boost,
                t=float(boost_t),
                task=str(task_type),
                device=DEVICE,
                save_name=str(BASE_DIR / "finetuned"),
                aux_bce_lambda=float(aux_bce),
                pos_weight=float(pos_weight) if pos_weight > 0 else None,
            )
            st.success("Training done!")
            torch.save(model.state_dict(), str(DATA_DIR / "finetuned_model.pt"))

elif __name__ == "__main__":
    main()

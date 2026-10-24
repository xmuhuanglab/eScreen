"""
eScreen GUI — Streamlit interface for prediction, interpretability, and training.
Usage:
    python -m escreen.serve --hostname 0.0.0.0 --port 8501

Interface surface (this module has no HTTP routes)
--------------------------------------------------
``serve.py`` is a **Streamlit** application, not a REST service: it defines no
``@app.route``, no FastAPI/Flask handler and no JSON-over-HTTP endpoint.  What it
exposes is a browser UI with three tabs, served by Streamlit's own websocket
protocol on ``--port`` (default 8501):

``Predict``
    Widgets: file uploader (``bed``/``tsv``/``txt``; headerless TSV whose first
    two columns are ``sequence`` and ``celltype``), ``Batch size`` number input
    (8-512, default 128), ``Predict`` button.  On submit the uploaded table is
    echoed with an extra ``prediction`` column and offered as
    ``predictions.tsv``.  Backed by ``eScreen_vX.predict``.

``Attribution``
    Widgets: free-text ``DNA sequence`` (truncated to 512 bp), free-text
    ``Cell type`` resolved by case-insensitive prefix matching against the known
    cell lines (exact single match is used directly, several matches produce a
    ``Suggestions`` selectbox, no match shows an error), ``Run attribution``
    button.  Output: a line chart of per-nucleotide importance.  Backed by
    Captum ``IntegratedGradients``.

``Train``
    Widgets: file uploader (TSV with ``sequence``, ``celltype``, ``label`` and an
    optional 4th ``score`` column), an expander with ``Epochs``,
    ``Learning rate``, ``Batch size``, ``Weight decay``, ``Task``
    (``reg``/``cls``), ``Boost resampling``, ``Boost temperature``,
    ``Early stop patience``, ``Aux BCE lambda``, ``Pos weight``, and a
    ``Start training`` button.  On submit it calls ``eScreen_vX.fit`` and writes
    ``data/finetuned_model.pt``.

CLI
---
``python -m escreen.serve --hostname 0.0.0.0 --port 8501`` re-executes this file
under ``streamlit run`` with ``_ESCREEN_SERVE=1`` in the environment; the
module-level block guarded by that flag is the actual application, and importing
the module has no side effects.

Relation to the manuscript
--------------------------
Supplementary Methods section 8 describes the *deployed* eScreen web server: a
Python web framework exposing RESTful API services, Tabix-indexed BED files for
CRE predictions and transcripts, BigWig tracks fetched on demand, and gene /
cCRE-ID / coordinate query modes over a 530,527-CRE compendium.  None of that is
implemented in this file.  This module is the local GUI for running the model
itself (predict, explain, fine-tune) and performs no indexed genomic retrieval;
section 8 and this module are different artefacts.
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

# Repository root and bundled data directory.  Every artefact (MEME file,
# checkpoint, Geneformer weights) is resolved relative to this, so the app can be
# launched from any working directory.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# One-hot channel order used by every model in this package: A, C, G, T.
BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def one_hot_encode(seq: str, max_len: int = 512) -> np.ndarray:
    """One-hot encode a DNA string into a fixed ``max_len`` x 4 uint8 matrix.

    Parameters
    ----------
    seq : str
        Nucleotide string.  Converted with ``.upper()`` and truncated to
        ``max_len``.
    max_len : int, default 512
        Number of positions to allocate - the eScreen input width
        (Supplementary Methods section 2.3.1: cCRE centre +/- 256 bp).

    Returns
    -------
    numpy.ndarray
        Shape ``(max_len, 4)``, dtype ``uint8``, channel order A, C, G, T.
        Short sequences are zero-padded at the tail, so only the first
        ``min(len(seq), max_len)`` rows are meaningful.

    Notes
    -----
    Characters outside A/C/G/T - ``N``, IUPAC codes, whitespace - leave an
    all-zero row rather than raising, matching the model's treatment of masked
    bases.  Silent truncation means a sequence longer than 512 bp is accepted
    without warning.
    """
    arr = np.zeros((max_len, 4), dtype=np.uint8)
    for j, b in enumerate(seq.upper()[:max_len]):
        if b in BASE_MAP:
            arr[j, BASE_MAP[b]] = 1
    return arr


@st.cache_resource
def init_model():
    """Build the eScreen model, its PWM front end and its cell-type embeddings.

    Cached with ``@st.cache_resource``, so the heavy objects are constructed once
    per process and shared across reruns and browser sessions.

    Returns
    -------
    model : escreen.model.eScreen_vX
        Model on ``DEVICE`` in ``eval()`` mode, warm-started from
        ``data/best_model.pt`` when that file exists.
    DEVICE : str
        ``'cuda'`` when available, otherwise ``'cpu'``.
    token2GFcelltype : dict
        Integer cell-type token -> cell-line name.
    GFcelltype2token : dict
        Cell-line name -> integer token (the inverse mapping).

    Notes
    -----
    * Both kernel tensors come from ``data/consensus_pwms.meme`` through
      ``escreen.motif.load_pwm_from_meme_c(..., max_length=35)``, i.e. the
      forward and reverse-complement PWM blocks padded to 35 bp as described in
      Supplementary Methods section 2.3.6.  Note that
      ``eSCREEN_backbone(filter_type='pwm')`` reads ``kwargs['kernel_fwd']``
      twice, so the reverse-complement tensor computed here is currently ignored
      and the 1,274 motif channels are the 637 forward PWMs duplicated rather
      than 637 forward + 637 reverse - a known mismatch with the manuscript (see
      the note in ``escreen/model.py``).  This function still builds and passes
      both, so fixing the backbone needs no change here.
    * Cell-type conditioning uses the Geneformer embeddings of
      ``data/DepMap_CellLine_embedding.pkl`` (mean-pooled DepMap bulk RNA-seq,
      Supplementary Methods section 2.3.5) plus the Geneformer V1-10M model and
      its 30M gene dictionaries under ``data/geneformer/``.
    * In the ``'pwm'`` branch the Conv1d width is ``2 * len(kernel_fwd)``
      (1,274) and the trainable projection goes to ``d_model`` (256), so
      ``num_filters=256`` does not size the motif layer at all; it is the
      ``conv``/``emb`` branches that consume it.
    * ``load_state_dict(..., strict=False)`` tolerates missing and extra keys
      silently, so a mismatched checkpoint degrades to partly random weights with
      no warning.  Inspect the returned ``_IncompatibleKeys`` if that matters.
    * All ``freeze_*`` flags are False, so the loaded model stays fully trainable:
      right for the Train tab, slower than a frozen encoder for pure inference.
    * ``DEVICE`` is chosen here, not by the caller, and is returned so the rest of
      the app can move tensors consistently.
    """
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_model = 256

    from escreen import eSCREEN_backbone, eScreen_vX
    from escreen.motif import load_pwm_from_meme_c
    from escreen.CellEmb import Geneformer

    # Motif kernels: forward and reverse-complement PWMs padded to 35 bp - the
    # fixed Conv1d weights of Supplementary Methods section 2.3.6.
    meme_file = str(DATA_DIR / "consensus_pwms.meme")
    motifs_f, motifs_r, _, _ = load_pwm_from_meme_c(meme_file, max_length=35)
    kernel_fwd = torch.tensor(motifs_f, dtype=torch.float)
    # NOTE (documented, not fixed): the backbone currently reads kernel_fwd for
    # both strands, so kernel_rev is built but never consumed; see the note in
    # eSCREEN_backbone's 'pwm' branch.
    kernel_rev = torch.tensor(motifs_r, dtype=torch.float)

    # Geneformer cell-type embeddings (Supplementary Methods section 2.3.5) and
    # the name <-> token maps used to condition every forward pass.
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

    # filter_type='pwm' selects the motif-initialised input stage: the Conv1d
    # weights are assigned from the PWM tensor and frozen inside the backbone,
    # so only the following 1274 -> d_model projection is trained.
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

    # Optional warm start from the shipped checkpoint; strict=False tolerates
    # architecture drift instead of failing.
    ckpt = DATA_DIR / "best_model.pt"
    if ckpt.exists():
        model.load_state_dict(
            torch.load(str(ckpt), map_location=DEVICE, weights_only=False),
            strict=False,
        )

    model.eval()
    return model, DEVICE, token2GFcelltype, GFcelltype2token


def df_to_data(df: pd.DataFrame, ct2idx: dict, seq_len: int = 512):
    """Convert a sequence/cell-type table into the dict the model consumes.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have ``sequence`` and ``celltype`` columns.
    ct2idx : dict
        Cell-line name -> integer token.  Mutated in place: unseen names are
        appended with index ``len(ct2idx)``.
    seq_len : int, default 512
        One-hot width per sequence.

    Returns
    -------
    data : dict
        ``{'sequence': ndarray (n, seq_len, 4) uint8, 'cell_type': ndarray (n,)
        int32, 'length': n}`` - exactly the layout ``eScreen_vX.predict`` and
        ``.fit`` index into.
    ct2idx : dict
        The (possibly extended) name -> token mapping.

    Notes
    -----
    Both call sites pass a fresh ``{}``, so tokens are assigned in
    first-appearance order instead of matching the Geneformer token ids: the
    cell-type conditioning is then arbitrary unless the uploaded names happen to
    line up.  Pass ``token2GFcelltype`` to reuse the pretrained mapping.
    """
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
    """Integrated Gradients attributions, reduced to one score per nucleotide.

    Implements the attribution of Supplementary Methods section 3.2: all-zero
    baseline, Gauss-Legendre numerical integration with 30 steps, then the
    completeness step (attribution multiplied element-wise by the input).

    Parameters
    ----------
    model : eScreen_vX
        Model whose ``forward(x, ct)`` returns one scalar per sample.  Set to
        ``eval()`` here regardless of the caller's state.
    seqs : numpy.ndarray
        One-hot input, shape ``(n, 512, 4)``; cast to float32 on ``device``.
    ct_indices : numpy.ndarray
        Cell-type tokens, shape ``(n,)``, passed as
        ``additional_forward_args``.
    device : str
        Torch device.
    batch_size : int, default 8
        Sequences per IG call; the cost is ``n_steps`` forward/backward passes
        per batch.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 512)``, float32.  Each entry is the sum over the four
        nucleotide channels of ``attribution * input``; because the input is
        one-hot, only the channel of the present base contributes, so the value
        is that base's signed importance.

    Notes
    -----
    * The convergence delta returned by Captum is discarded: nothing checks that
      the 30-step Gauss-Legendre approximation converged.
    * No CRE-internal normalisation is applied here.  The paper's
      ``norm_mean`` (motif-interval mean divided by whole-CRE mean,
      Supplementary Methods section 3.3.2) belongs downstream.
    * ``eScreen_vX.forward`` accepts ``(x, ct, return_cls, return_emb)``, so IG
      explains the default regression output (``return_cls=False``).
    * Memory scales with ``batch_size`` because gradients are retained on the
      expanded interpolation batch.
    """
    from captum.attr import IntegratedGradients
    model.eval()
    ig = IntegratedGradients(model.forward)
    all_attr = []
    n = len(seqs)
    for i in range(0, n, batch_size):
        bx = torch.tensor(seqs[i:i + batch_size], dtype=torch.float, device=device).requires_grad_()
        bc = torch.tensor(ct_indices[i:i + batch_size], dtype=torch.int, device=device)
        # IG from the all-zero baseline with Gauss-Legendre quadrature over 30
        # steps: Supplementary Methods section 3.2.
        bl = torch.zeros_like(bx)
        attr, _ = ig.attribute(
            bx, baselines=bl, additional_forward_args=bc,
            return_convergence_delta=True, method="gausslegendre", n_steps=30,
        )
        # Completeness step: attribution x input, summed over the 4 channels.
        score = (attr.detach().cpu().numpy() * bx.detach().cpu().numpy()).sum(axis=-1)
        all_attr.append(score)
    return np.concatenate(all_attr, axis=0)


def main():
    """CLI entry point: relaunch this file under ``streamlit run``.

    The function does not serve anything itself.  It re-invokes the current
    interpreter with ``-m streamlit run <this file>``, forwarding the host and
    port and forcing headless mode, and exports ``_ESCREEN_SERVE=1`` so that the
    module-level app block executes in the child process.  This double-launch is
    what lets ``python -m escreen.serve`` behave like a single command.

    Parameters
    ----------
    (none)
        Parsed from ``argv``: ``--hostname`` (default ``0.0.0.0``) and
        ``--port`` (default ``8501``).

    Returns
    -------
    None
        Blocks until the Streamlit child process exits.

    Notes
    -----
    ``subprocess.run`` is called without ``check=True``, so a failing Streamlit
    child leaves the parent with a zero exit status - scripted callers cannot
    detect a crashed server from the return code.
    """
    parser = argparse.ArgumentParser(description="eScreen GUI")
    parser.add_argument("--hostname", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    env = os.environ.copy()
    env["_ESCREEN_SERVE"] = "1"
    # Re-exec under `streamlit run`; the environment flag below makes the
    # module-level block execute in the child process only.
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", __file__,
        "--server.address", args.hostname,
        "--server.port", str(args.port),
        "--server.headless", "true",
    ], env=env)


# ---------------------------------------------------------------------------
# Streamlit app body.  Executes only in the child process started by main(),
# which is why importing escreen.serve has no side effects.  Structure: one
# cached model (init_model), then three tabs - Predict, Attribution, Train.
# Streamlit reruns this whole block on every widget interaction.
# ---------------------------------------------------------------------------
if os.environ.get("_ESCREEN_SERVE") == "1":
    st.set_page_config(page_title="eScreen", layout="wide")
    st.title("eScreen — functional CRE screening")

# Build the model once per process: the checkpoint load and Geneformer init are
# the expensive part, so they must not re-run per widget.
    model, DEVICE, token2celltype, celltype2token = init_model()
    known_celltypes = sorted(celltype2token.keys())

# Three tabs, one per entry point of the package: inference, attribution and
# fine-tuning.
    tab_pred, tab_attr, tab_train = st.tabs(["Predict", "Attribution", "Train"])

    # ---- Predict: (sequence, celltype) rows -> activity score -------------
    with tab_pred:
        st.subheader("Predict regulatory activity")
        pred_file = st.file_uploader(
            "Upload BED file (TSV, columns: sequence, celltype)",
            type=['bed', 'tsv', 'txt'], key='pred',
        )
        pred_batch = st.number_input("Batch size", 8, 512, 128, key='pred_bs')
        pred_btn = st.button("Predict", key='pred_btn')

        if pred_btn and pred_file is not None:
            # Headerless TSV: the first two columns are taken as
            # sequence/celltype, everything else in the file is ignored.
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

    # ---- Attribution: Integrated Gradients over one 512 bp sequence ------
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
                # Case-insensitive prefix match against the known cell lines;
                # ambiguous prefixes must be disambiguated before running.
                # 'attr_ct' stays None when nothing matches, which disables the
                # button's effect below.
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

    # ---- Train: fine-tune the loaded checkpoint on a labelled table ------
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
            # Layout: sequence, celltype, label[, score].  'score' (continuous)
            # becomes the regression target; 'label' (binary) drives the
            # auxiliary BCE head and the boosted sampler.
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
            # Regression target: the continuous score when present, otherwise
            # the binary label cast to float.
            # has_binary gates the auxiliary classification target: a label
            # column with more than two distinct values cannot supervise BCE and
            # is then used only through 'y'.
            data['y'] = df['score'].values.astype(np.float32) if has_score else df['label'].values.astype(np.float32)
            has_binary = df['label'].dropna().nunique() <= 2
            if has_binary:
                data['label'] = df['label'].values.astype(np.float32)

            opt = torch.optim.AdamW(model.parameters(), lr=float(train_lr), weight_decay=float(train_wd))

            model.train()
            # fit() owns the boosting sampler, the smooth-L1 + auxiliary BCE
            # composite loss, per-epoch checkpointing and early stopping.
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
            # Persist the fine-tuned weights next to the shipped checkpoint.
            torch.save(model.state_dict(), str(DATA_DIR / "finetuned_model.pt"))

elif __name__ == "__main__":
    main()

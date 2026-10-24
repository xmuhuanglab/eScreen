
#@title Huggingface Pretrained Wrapper

"""

This is script is a simple HuggingFace wrapper around a HyenaDNA model, to enable a one click example
of how to load the pretrained weights and get embeddings.

It will instantiate a HyenaDNA model (model class is in the `standalone_hyenadna.py`), and handle the downloading of pretrained weights from HuggingFace.

Check out the colab notebook for a simpler and more complete walk through of how to use HyenaDNA with pretrained weights.

"""


import json
import os
import subprocess
import torch
# import transformers
from transformers import PreTrainedModel
import re
from standalone_hyenadna import HyenaDNAModel
from standalone_hyenadna import CharacterTokenizer


# helper 1
def inject_substring(orig_str):
    """Hack to handle matching keys between models trained with and without
    gradient checkpointing."""

    # modify for mixer keys
    pattern = r"\.mixer"
    injection = ".mixer.layer"

    modified_string = re.sub(pattern, injection, orig_str)

    # modify for mlp keys
    pattern = r"\.mlp"
    injection = ".mlp.layer"

    modified_string = re.sub(pattern, injection, modified_string)

    return modified_string

# helper 2
def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """Loads pretrained (backbone only) weights into the scratch state dict."""

    # loop thru state dict of scratch
    # find the corresponding weights in the loaded model, and set it

    # need to do some state dict "surgery"
    for key, value in scratch_dict.items():
        if 'backbone' in key:
            # the state dicts differ by one prefix, '.model', so we add that
            key_loaded = 'model.' + key
            # breakpoint()
            # need to add an extra ".layer" in key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except:
                raise Exception('key mismatch in the state dicts!')

    # scratch_dict has been updated
    return scratch_dict

class HyenaDNAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    base_model_prefix = "hyenadna"

    def __init__(self, config):
        """Leave the base-class initialisation to the subclass.

        ``PreTrainedModel.__init__`` is deliberately not called: subclasses attach
        their own ``self.model`` after construction, so there is nothing to set up
        yet.  Constructing this class on its own yields an object with no ``model``
        attribute.
        """
        pass

    def forward(self, input_ids, **kwargs):
        """Run the underlying HyenaDNA model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token ids, ``[B, L]``.
        **kwargs
            Forwarded unchanged to the backbone.

        Returns
        -------
        torch.Tensor
            Backbone output: logits when ``use_head`` was set, hidden states
            otherwise.
        """
        return self.model(input_ids, **kwargs)

    @classmethod
    def from_pretrained(cls,
                        path,
                        model_name,
                        download=False,
                        config=None,
                        device='cpu',
                        use_head=False,
                        n_classes=2,
                      ):
        # first check if it is a local path
        """Load a HyenaDNA checkpoint, locally or from HuggingFace.

        Parameters
        ----------
        path : str
            Directory holding (or to hold) the checkpoint.
        model_name : str
            Variant name, e.g. ``'tiny-1k-d256'``.
        download : bool, default False
            When the local directory is absent, clone the weights from
            ``https://huggingface.co/LongSafari/<model_name>`` using git-lfs.
        config : dict, optional
            Model configuration; read from ``config.json`` when omitted.
        device : str, default 'cpu'
            Device passed to ``torch.load``.
        use_head : bool, default False
            Whether to attach the classification head.
        n_classes : int, default 2
            Number of classes for that head.

        Returns
        -------
        HyenaDNAModel
            The model with pretrained weights applied.

        Notes
        -----
        Loading takes two steps because the checkpoint stores its keys under a
        ``model.`` prefix that the freshly built module does not use:
        :func:`load_weights` rewrites the prefixes and, when the checkpoint was
        trained with gradient checkpointing, injects an extra ``.layer`` segment.
        A key that cannot be resolved raises rather than leaving a randomly
        initialised weight in place.

        When ``download`` is true the destination directory is removed first
        (``rm -rf`` through a shell), which is destructive - point ``path`` at a
        scratch directory.
        """
        pretrained_model_name_or_path = os.path.join(path, model_name)
        if os.path.isdir(pretrained_model_name_or_path) and download == False:
            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))
        else:
            hf_url = f'https://huggingface.co/LongSafari/{model_name}'

            subprocess.run(f'rm -rf {pretrained_model_name_or_path}', shell=True)
            command = f'mkdir -p {path} && cd {path} && git lfs install && git clone {hf_url}'
            subprocess.run(command, shell=True)

            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))

        scratch_model = HyenaDNAModel(**config, use_head=use_head, n_classes=n_classes)  # the new model format
        loaded_ckpt = torch.load(
            os.path.join(pretrained_model_name_or_path, 'weights.ckpt'),
            map_location=torch.device(device),
            weights_only=False
        )

        # need to load weights slightly different if using gradient checkpointing
        if config.get("checkpoint_mixer", False):
            checkpointing = config["checkpoint_mixer"] == True or config["checkpoint_mixer"] == True
        else:
            checkpointing = False

        # grab state dict from both and load weights
        state_dict = load_weights(scratch_model.state_dict(), loaded_ckpt['state_dict'], checkpointing=checkpointing)

        # scratch model has now been updated
        scratch_model.load_state_dict(state_dict)
        print("Loaded pretrained weights ok!")
        return scratch_model




####################################################################################################




"""# Inference (450k to 1M tokens)!

If all you're interested in is getting embeddings on long DNA sequences
(inference), then we can do that right here in Colab!


*   We provide an example how to load the weights from Huggingface.
*   On the free tier, which uses a
T4 GPU w/16GB of memory, we can process 450k tokens / nucleotides.
*   For processing 1M tokens, you'll need an A100, which Colab offers as a paid tier.
*   (Don't forget to run the entire notebook above too)

--

To pretrain or fine-tune the 1M long sequence model (8 layers, d_model=256),
you'll need 8 A100s 80GB, and all that code is in the main repo!
"""

#@title Single example
import json
import os
import subprocess
# import transformers
from transformers import PreTrainedModel

def inference_single():

    '''
    this selects which backbone to use, and grabs weights/ config from HF
    4 options:
      'hyenadna-tiny-1k-seqlen'   # fine-tune on colab ok
      'hyenadna-small-32k-seqlen'
      'hyenadna-medium-160k-seqlen'  # inference only on colab
      'hyenadna-medium-450k-seqlen'  # inference only on colab
      'hyenadna-large-1m-seqlen'  # inference only on colab
    '''

    # you only need to select which model to use here, we'll do the rest!
    pretrained_model_name = 'hyenadna-small-32k-seqlen'

    max_lengths = {
        'hyenadna-tiny-1k-seqlen': 1024,
        'hyenadna-small-32k-seqlen': 32768,
        'hyenadna-medium-160k-seqlen': 160000,
        'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
        'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
    }

    max_length = max_lengths[pretrained_model_name]  # auto selects

    # data settings:
    use_padding = True
    rc_aug = False  # reverse complement augmentation
    add_eos = False  # add end of sentence token

    # we need these for the decoder head, if using
    use_head = False
    n_classes = 2  # not used for embeddings only

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one in None
    backbone_cfg = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # instantiate the model (pretrained here)
    if pretrained_model_name in ['hyenadna-tiny-1k-seqlen',
                                 'hyenadna-small-32k-seqlen',
                                 'hyenadna-medium-160k-seqlen',
                                 'hyenadna-medium-450k-seqlen',
                                 'hyenadna-large-1m-seqlen']:
        # use the pretrained Huggingface wrapper instead
        model = HyenaDNAPreTrainedModel.from_pretrained(
            './checkpoints',
            pretrained_model_name,
            download=True,
            config=backbone_cfg,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )

    # from scratch
    elif pretrained_model_name is None:
        model = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes)

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='left', # since HyenaDNA is causal, we pad on the left
    )

    #### Single embedding example ####

    # create a sample 450k long, prepare
    sequence = 'ACTG' * int(max_length/4)
    tok_seq = tokenizer(sequence)
    tok_seq = tok_seq["input_ids"]  # grab ids

    # place on device, convert to tensor
    tok_seq = torch.LongTensor(tok_seq).unsqueeze(0)  # unsqueeze for batch dim
    tok_seq = tok_seq.to(device)

    # prep model and forward
    model.to(device)
    model.eval()
    with torch.inference_mode():
        embeddings = model(tok_seq)

    print(embeddings.shape)  # embeddings here!

# # uncomment to run! (to get embeddings)
#inference_single()


# to run this, just call:
    # python huggingface.py

import gc

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_attacks import get_embedding_matrix, get_embeddings

alpha = 0.5     #TODO: 0.9
refuse_tokens = [
    "I", "As", "Sorry", "unethical", "illegal", "cannot", "However", "But", "must", "please"
]

def token_gradients(model, tokenizer, input_ids, input_slice, target_slice, loss_slice):
    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    accept_loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)

    # 拒绝损失token
    refuse_loss = 0
    length_of_refuse = target_slice.stop - target_slice.start
    for token in refuse_tokens:
        refuse_token = tokenizer(token, add_special_tokens=False).input_ids
        padding_refuse_token = [
            refuse_token * (length_of_refuse // len(refuse_token))
        ]
        padding_refuse_token = padding_refuse_token[0]
        if length_of_refuse % len(refuse_token) != 0:
            padding_refuse_token.extend(
                refuse_token[:length_of_refuse % len(refuse_token)]
            )

        refuse_target = torch.tensor(
            padding_refuse_token,
            device=model.device, dtype=torch.long
        )
        refuse_loss += nn.CrossEntropyLoss()(logits[0,loss_slice,:], refuse_target)

    refuse_loss = - refuse_loss / len(refuse_tokens)

    loss = alpha * accept_loss + (1 - alpha) * refuse_loss 
    
    loss.backward()
    
    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    
    return grad

def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks


def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        # decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        tokens = tokenizer.convert_ids_to_tokens(control_cand[i], skip_special_tokens=True)
        clean_tokens = [token[1:] for token in tokens if token not in tokenizer.all_special_tokens]
        decoded_str = ' '.join(clean_tokens)

        # print(f"Control {i}: {decoded_str}, {len(tokenizer(decoded_str, add_special_tokens=False).input_ids)}, {len(control_cand[i])}")
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
    return cands


def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    
    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if not(test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), " 
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids ; gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        return logits
    

def forward(*, model, input_ids, attention_mask, batch_size=512):

    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        
        batch_input_ids = input_ids[i:i+batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+batch_size]
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask
    
    return torch.cat(logits, dim=0)

def target_loss(model, tokenizer, logits, ids, target_slice):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    accept_loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,target_slice])

    # 拒绝损失token
    refuse_loss = 0
    length_of_refuse = target_slice.stop - target_slice.start
    for token in refuse_tokens:
        refuse_token = tokenizer(token, add_special_tokens=False).input_ids
        padding_refuse_token = [
            refuse_token * (length_of_refuse // len(refuse_token))
        ]
        padding_refuse_token = padding_refuse_token[0]
        if length_of_refuse % len(refuse_token) != 0:
            padding_refuse_token.extend(
                refuse_token[:length_of_refuse % len(refuse_token)]
            )

        refuse_target = torch.tensor(
            padding_refuse_token,
            device=model.device, dtype=torch.long
        )

        # add the batched dimension for refuse target
        refuse_target = refuse_target.unsqueeze(0).repeat(ids.shape[0], 1)
        
        # Ensure refuse_target has the same shape as ids[:, target_slice]
        refuse_target = refuse_target[:, :ids[:, target_slice].shape[1]]

        refuse_loss += crit(logits[:,loss_slice,:].transpose(1,2), refuse_target)

    refuse_loss = - refuse_loss / len(refuse_tokens)
    
    loss = alpha * accept_loss + (1-alpha) * refuse_loss

    return loss.mean(dim=-1)


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    # print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs
        ).to(device).eval()
    
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'llama-3' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

























from __future__ import annotations

import importlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.projectors.modal_projector import ModalProjector



MODALITY_IDS = {"fmri": 1, "eeg": 2, "meg": 3}


def _load_llm(llm_path: str, torch_dtype, device_map: str):






    from transformers import AutoConfig
    try:
        model = AutoModelForCausalLM.from_pretrained(
            llm_path, dtype=torch_dtype, device_map=device_map,
            trust_remote_code=True,
        )
        return model
    except ValueError:
        pass


    cfg = AutoConfig.from_pretrained(llm_path, trust_remote_code=True)
    archs = getattr(cfg, "architectures", [])
    if not archs:
        raise RuntimeError(f"Cannot determine model class for {llm_path}")
    arch_cls = getattr(importlib.import_module("transformers"), archs[0])
    full_model = arch_cls.from_pretrained(
        llm_path, dtype=torch_dtype, device_map=device_map,
        trust_remote_code=True,
    )

    if hasattr(full_model, "thinker"):
        return full_model.thinker
    return full_model


def _resolve_hidden_size(model) -> int:


    hs = getattr(model.config, "hidden_size", None)
    if hs:
        return hs

    try:
        hs = model.config.text_config.hidden_size
        if hs:
            return hs
    except AttributeError:
        pass

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "in_features"):
        return model.lm_head.in_features

    try:
        return model.model.embed_tokens.embedding_dim
    except AttributeError:
        pass
    raise ValueError("Cannot infer LLM hidden_size. Check the model config.")


class BCILanguageModel(nn.Module):



















    def __init__(
        self,
        llm_path: str,
        d_bci: int = 1024,
        freeze_llm: bool = True,
        torch_dtype=torch.bfloat16,
        device_map: str = "auto",
    ):
        super().__init__()


        self.llm = _load_llm(llm_path, torch_dtype, device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        d_llm: int = _resolve_hidden_size(self.llm)


        self.set_llm_frozen(freeze_llm)


        self.projector = ModalProjector(d_model=d_bci, d_llm=d_llm)



        self.modality_embed = nn.Embedding(4, d_llm)
        nn.init.trunc_normal_(self.modality_embed.weight, std=0.02)

        self.d_llm = d_llm


        self.llm_dtype = next(
            p.dtype for p in self.llm.parameters()
            if p.device.type != "meta"
        )




        _first_dev = None
        for p in self.llm.parameters():
            if p.device.type != "meta":
                _first_dev = p.device
                break
        if _first_dev is not None:
            self.projector     = self.projector.to(_first_dev)
            self.modality_embed = self.modality_embed.to(_first_dev)







        with torch.no_grad():
            try:
                _sample_ids  = torch.zeros(1, 8, dtype=torch.long,
                                           device=_first_dev or "cpu")
                _sample_emb  = self._get_llm_embeddings(_sample_ids).float()
                _embed_std   = _sample_emb.std().item()
            except Exception:
                _embed_std = 0.01
        self._bci_embed_scale: float = max(_embed_std, 1e-6)



    def set_llm_frozen(self, frozen: bool) -> None:
        for p in self.llm.parameters():
            p.requires_grad = not frozen

    def apply_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.05,
    ) -> None:




        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("peft is required for LoRA. Run: pip install peft")

        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_cfg)
        self.llm.print_trainable_parameters()



    def _get_llm_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:

        llm = self.llm

        if hasattr(llm, "base_model") and hasattr(llm.base_model, "model"):
            return llm.base_model.model.embed_tokens(input_ids)

        if hasattr(llm, "model") and hasattr(llm.model, "embed_tokens"):
            return llm.model.embed_tokens(input_ids)
        raise AttributeError(
            f"Cannot locate embed_tokens in {type(llm).__name__}. "
            "Add a custom branch to _get_llm_embeddings."
        )

    def _build_inputs(
        self,
        bci_embeddings: Dict[str, torch.Tensor],
        prompt_ids: torch.Tensor,
        answer_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:








        device = prompt_ids.device
        B = prompt_ids.shape[0]
        parts_embeds = []
        parts_mask   = []
        parts_labels = []


        proj_w      = self.projector.proj[0].weight
        proj_device = proj_w.device
        proj_dtype  = proj_w.dtype


        for mod_name, mod_id in [("fmri", 1), ("meg", 3), ("eeg", 2)]:
            if mod_name not in bci_embeddings:
                continue
            emb = bci_embeddings[mod_name]
            if not isinstance(emb, torch.Tensor):
                continue

            emb = emb.to(device=proj_device, dtype=proj_dtype)
            proj = self.projector(emb)

            type_bias = self.modality_embed(
                torch.tensor(mod_id, device=proj_device)
            )
            proj = proj + type_bias





            proj = proj * self._bci_embed_scale

            N = proj.shape[1]
            parts_embeds.append(proj)
            parts_mask.append(torch.ones(B, N, dtype=torch.long, device=device))
            parts_labels.append(torch.full((B, N), -100, dtype=torch.long, device=device))


        prompt_pad_mask = (prompt_ids != self.tokenizer.pad_token_id).long()
        prompt_emb = self._get_llm_embeddings(
            prompt_ids.clamp(min=0)
        ).to(device=proj_device, dtype=proj_dtype)

        parts_embeds.append(prompt_emb)
        parts_mask.append(prompt_pad_mask)
        parts_labels.append(torch.full_like(prompt_ids, -100))


        if answer_ids is not None:
            answer_pad_mask = (answer_ids != -100).long()

            answer_emb = self._get_llm_embeddings(
                answer_ids.clamp(min=0)
            ).to(device=proj_device, dtype=proj_dtype)

            parts_embeds.append(answer_emb)
            parts_mask.append(answer_pad_mask)

            parts_labels.append(answer_ids)

        inputs_embeds  = torch.cat(parts_embeds, dim=1)
        attention_mask = torch.cat(parts_mask,   dim=1)
        labels         = torch.cat(parts_labels, dim=1)

        return inputs_embeds, attention_mask, labels

    def forward(
        self,
        bci_embeddings: Dict[str, torch.Tensor],
        prompt_ids: torch.Tensor,
        answer_ids: torch.Tensor,
    ) -> torch.Tensor:











        inputs_embeds, attention_mask, labels = self._build_inputs(
            bci_embeddings, prompt_ids, answer_ids
        )


        inputs_embeds = inputs_embeds.to(dtype=self.llm_dtype)

        attention_mask = attention_mask.to(device=inputs_embeds.device)
        labels         = labels.to(device=inputs_embeds.device)











        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )





        logits       = out.logits
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous().to(
            device=shift_labels.device, dtype=torch.float32
        )
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    @torch.no_grad()
    def generate(
        self,
        bci_embeddings: Dict[str, torch.Tensor],
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        **generate_kwargs,
    ) -> List[str]:






        inputs_embeds, attention_mask, _ = self._build_inputs(
            bci_embeddings, prompt_ids, answer_ids=None
        )
        inputs_embeds = inputs_embeds.to(dtype=self.llm_dtype)
        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )



        texts = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )
        return texts

    def trainable_parameters(self):

        return (p for p in self.parameters() if p.requires_grad)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

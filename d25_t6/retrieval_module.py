import copy
import math
import string
from typing import Any
import os
import ast
import pandas as pd
import numpy as np
import torch
from lightning import pytorch as pl
from transformers import RobertaTokenizer, RobertaModel
from prettytable import PrettyTable

from d25_t6.passt import CutInputIntoSegmentsWrapper, PaSSTSNoOverlapWrapper
from d25_t6.losses import AlignmentContrastiveLoss, ContrastiveLoss, l2norm


class AudioRetrievalModel(pl.LightningModule):

    def __init__(
            self,
            **kwargs
    ):

        super().__init__()
        self.save_hyperparameters(kwargs)

        # audio encoder
        self.audio_embedding_model = CutInputIntoSegmentsWrapper(
            PaSSTSNoOverlapWrapper(
                s_patchout_t=kwargs['s_patchout_t'],
                s_patchout_f=kwargs['s_patchout_f']
            ),
            max_input_length=10*32000,
            segment_length=10*32000,
            hop_size=10*32000
        )
        self.audio_projection = torch.nn.Linear(768, 1024)

        # text encoder
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.text_embedding_model = RobertaModel.from_pretrained(
            'roberta-base' if kwargs['roberta_base'] else 'roberta-large',
            add_pooling_layer=False,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            output_hidden_states=False
        )
        self.text_projection = torch.nn.Linear(768 if kwargs['roberta_base'] else 1024, 1024)

        # temperature parameter
        initial_tau = torch.zeros((1,)) + kwargs['initial_tau']
        self.tau = torch.nn.Parameter(initial_tau, requires_grad=kwargs['tau_trainable'])

        # Intra-Modal Alignment parameters
        self.enable_intra_modal_alignment = kwargs.get('enable_intra_modal_alignment', False)
        self.enable_matching_loss = kwargs.get('enable_matching_loss', False)
        self.enable_alignment_loss = kwargs.get('enable_alignment_loss', False)
        self.alignment_loss_weight = kwargs.get('alignment_loss_weight', 0.4)
        self.matching_loss_weight = kwargs.get('matching_loss_weight', 1.0)
        
        # Initialize loss functions if Intra-Modal Alignment is enabled
        if self.enable_intra_modal_alignment:
            delta = kwargs.get('delta', 0.2)
            measure = kwargs.get('measure', 'cosine')
            max_violation = kwargs.get('max_violation', True)
            aggregation = kwargs.get('aggregation', 'sum-max-sentences')
            sigma = kwargs.get('sigma', 0.0)
            
            if self.enable_alignment_loss:
                self.alignment_criterion = AlignmentContrastiveLoss(
                    margin=delta,
                    measure=measure,
                    max_violation=max_violation,
                    aggregation=aggregation
                )
            
            if self.enable_matching_loss:
                self.matching_criterion = ContrastiveLoss(
                    margin=delta,
                    measure=measure,
                    max_violation=max_violation,
                    sigma=sigma
                )

        self.validation_outputs = []

        self.kwargs = kwargs

        self.compile_model()

    def compile_model(self):
        """Apply torch.compile() if GPU is recent"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()  # Get current GPU device
            properties = torch.cuda.get_device_properties(device)
            if properties.major >= 7 and self.kwargs['compile'] == True:
                print("Compiling Models")
                self.text_embedding_model = torch.compile(self.text_embedding_model)
                self.audio_embedding_model.model.model = torch.compile(self.audio_embedding_model.model.model)

    def forward(self, batch) -> Any:

        # embed audio & text
        text_embeddings = self.forward_text(batch)
        audio_embeddings = self.forward_audio(batch)

        return audio_embeddings, text_embeddings
    
    def forward_audio_with_local(self, batch):
        """
        Forward audio and return both global and local features
        Returns:
            global_features: (batch, dim) - global audio features
            local_features: (batch, seq_len, dim) - local audio features
            audio_lengths: list of actual sequence lengths
        """
        audio_embeddings = self.audio_embedding_model(batch['audio'].mean(1))  # (batch, seq_len, 768)
        
        # Get local features (before projection)
        local_features = audio_embeddings  # (batch, seq_len, 768)
        
        # Calculate actual lengths based on duration
        audio_lengths = []
        for i, duration in enumerate(batch['duration']):
            if duration <= 10:
                audio_lengths.append(1)
            elif duration <= 20:
                audio_lengths.append(min(2, audio_embeddings.shape[1]))
            else:
                audio_lengths.append(audio_embeddings.shape[1])
        
        # Aggregate for global features
        aggregated = []
        for i, duration in enumerate(batch['duration']):
            if duration <= 10:
                aggregated.append(audio_embeddings[i, 0])
            elif duration <= 20:
                aggregated.append(audio_embeddings[i, :2].mean(-2))
            else:
                aggregated.append(audio_embeddings[i].mean(-2))
        
        global_features = torch.stack(aggregated)
        global_features = self.audio_projection(global_features)  # (batch, 1024)
        global_features = torch.nn.functional.normalize(global_features, p=2, dim=-1)
        
        # Project local features
        batch_size, seq_len, dim = local_features.shape
        local_features = local_features.reshape(-1, dim)  # (batch*seq_len, 768)
        local_features = self.audio_projection(local_features)  # (batch*seq_len, 1024)
        local_features = local_features.reshape(batch_size, seq_len, -1)  # (batch, seq_len, 1024)
        local_features = torch.nn.functional.normalize(local_features, p=2, dim=2)
        
        return global_features, local_features, audio_lengths
    
    def forward_text_with_local(self, batch):
        """
        Forward text and return both global and local features
        Returns:
            global_features: (batch, dim) - global text features
            local_features: (batch, seq_len, dim) - local text features
            text_lengths: list of actual sequence lengths
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        captions = []
        for i, b in enumerate([c[0] for c in batch['captions']]):
            if not (type(b) == str):
                print(b)
                b = b[0]
            captions.append(b.lower().translate(str.maketrans('', '', string.punctuation)))

        tokenized = self.tokenizer(
            captions,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            max_length=32,
            truncation=True
        )

        token_embeddings = self.text_embedding_model(
            input_ids=tokenized['input_ids'].to(device),
            attention_mask=tokenized['attention_mask'].to(device)
        )[0]  # (batch, seq_len, dim)
        
        # Get local features (before projection)
        local_features = token_embeddings  # (batch, seq_len, 768 or 1024)
        
        # Calculate actual lengths
        text_lengths = tokenized['attention_mask'].sum(dim=1).cpu().tolist()
        
        # Global features (CLS token)
        global_features = token_embeddings[:, 0, :]  # (batch, dim)
        global_features = self.text_projection(global_features)  # (batch, 1024)
        global_features = torch.nn.functional.normalize(global_features, p=2, dim=-1)
        
        # Project local features
        batch_size, seq_len, dim = local_features.shape
        local_features = local_features.reshape(-1, dim)  # (batch*seq_len, dim)
        local_features = self.text_projection(local_features)  # (batch*seq_len, 1024)
        local_features = local_features.reshape(batch_size, seq_len, -1)  # (batch, seq_len, 1024)
        local_features = torch.nn.functional.normalize(local_features, p=2, dim=2)
        
        return global_features, local_features, text_lengths

    def forward_audio(self, batch):

        audio_embeddings = self.audio_embedding_model(batch['audio'].mean(1)) # forward

        # mask embeddings from padded empty audio parts
        aggregated = []
        for i, duration in enumerate(batch['duration']):
            if duration <= 10:
                aggregated.append(audio_embeddings[i, 0])
            elif duration <= 20:
                aggregated.append(audio_embeddings[i, :2].mean(-2))
            else:
                aggregated.append(audio_embeddings[i].mean(-2))

        audio_embeddings = torch.stack(aggregated)
        audio_embeddings = self.audio_projection(audio_embeddings) # project to same dimension
        audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=-1) # normalize
        return audio_embeddings

    def forward_text(self, batch):

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        captions = []
        for i, b in enumerate([c[0] for c in batch['captions']]):
            if not (type(b) == str):
                print(b)
                b = b[0]
            captions.append(b.lower().translate(str.maketrans('', '', string.punctuation)))

        tokenized = self.tokenizer(
            captions,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            max_length=32,
            truncation=True
        )

        token_embeddings = self.text_embedding_model(
            input_ids=tokenized['input_ids'].to(device),
            attention_mask=tokenized['attention_mask'].to(device)
        )[0]
        # select first token of sequence
        sentence_features = token_embeddings[:, 0, :]
        # project
        sentence_features = self.text_projection(sentence_features)
        # normalize
        sentence_features = torch.nn.functional.normalize(sentence_features, p=2, dim=-1)

        return sentence_features

    def training_step(self, batch, batch_idx):

        self.lr_scheduler_step(batch_idx)

        # Get embeddings based on whether Intra-Modal Alignment is enabled
        if self.enable_intra_modal_alignment:
            # Get global and local features
            audio_global, audio_local, audio_lengths = self.forward_audio_with_local(batch)
            text_global, text_local, text_lengths = self.forward_text_with_local(batch)
            
            # Use global features for main contrastive loss
            audio_embeddings = audio_global
            text_embeddings = text_global
        else:
            # Original behavior: only global features
            audio_embeddings, text_embeddings = self.forward(batch)

        # compute pairwise similarities
        C = torch.matmul(audio_embeddings, text_embeddings.T)

        # scale cosine similarities with temperature < 1
        # (otherwise $-1 <= C_{ij} <= 1$)
        C = C / torch.abs(self.tau)

        # compute P(a|t) and P(t|a)
        C_audio = torch.log_softmax(C, dim=0)
        C_text = torch.log_softmax(C, dim=1)

        # prediction target
        paths = np.array([hash(batch['dataset'][i] + batch['subset'][i] + p) for i, p in enumerate(batch['fname'])])
        I = torch.tensor(paths[None, :] == paths[:, None])

        # Main contrastive loss
        main_loss = -0.5 * (C_audio[torch.where(I)].mean() + C_text[torch.where(I)].mean())
        total_loss = main_loss
        
        # Add Intra-Modal Alignment losses if enabled
        if self.enable_intra_modal_alignment:
            # Handle duplicate paths: average features for duplicate paths
            unique_paths_dict = {}
            for i, p in enumerate(paths):
                if p not in unique_paths_dict:
                    unique_paths_dict[p] = []
                unique_paths_dict[p].append(i)
            
            # Get unique paths and their indices
            unique_paths_list = list(unique_paths_dict.keys())
            num_unique = len(unique_paths_list)
            
            # Initialize tensors for averaged features
            device = audio_global.device
            audio_global_unique = torch.zeros(num_unique, audio_global.size(1), device=device)
            text_global_unique = torch.zeros(num_unique, text_global.size(1), device=device)
            audio_local_unique = torch.zeros(num_unique, audio_local.size(1), audio_local.size(2), device=device)
            text_local_unique = torch.zeros(num_unique, text_local.size(1), text_local.size(2), device=device)
            audio_lengths_unique = []
            text_lengths_unique = []
            mask = torch.zeros(len(paths), dtype=torch.bool, device=device)
            
            # Average features for each unique path
            for unique_idx, path in enumerate(unique_paths_list):
                indices = unique_paths_dict[path]
                replacement_idx = indices[0]
                mask[replacement_idx] = True
                
                # Average global features
                audio_global_unique[unique_idx] = audio_global[indices].mean(0)
                text_global_unique[unique_idx] = text_global[indices].mean(0)
                
                # Average local features
                audio_local_unique[unique_idx] = audio_local[indices].mean(0)
                text_local_unique[unique_idx] = text_local[indices].mean(0)
                
                # Use average length (or max, depending on your preference)
                audio_lengths_unique.append(max([audio_lengths[i] for i in indices]))
                text_lengths_unique.append(max([text_lengths[i] for i in indices]))
            
            # Matching loss (global-to-global)
            if self.enable_matching_loss:
                matching_loss = self.matching_criterion(audio_global_unique, text_global_unique)
                total_loss = total_loss + matching_loss * self.matching_loss_weight
                self.log("train/matching_loss", matching_loss, batch_size=len(audio_embeddings), sync_dist=True)
            
            # Alignment loss (local-to-local)
            if self.enable_alignment_loss:
                alignment_loss = self.alignment_criterion(
                    audio_local_unique, 
                    text_local_unique, 
                    audio_lengths_unique, 
                    text_lengths_unique
                )
                total_loss = total_loss + alignment_loss * self.alignment_loss_weight
                self.log("train/alignment_loss", alignment_loss, batch_size=len(audio_embeddings), sync_dist=True)

        self.log("train/loss", total_loss, batch_size=len(audio_embeddings), sync_dist=True, prog_bar=True)
        self.log("train/main_loss", main_loss, batch_size=len(audio_embeddings), sync_dist=True)
        self.log('train/tau', torch.abs(self.tau), sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):

        audio_embeddings, text_embeddings = self.forward(batch)

        args = {
            'audio_embeddings': copy.deepcopy(audio_embeddings.detach()),
            'text_embeddings': copy.deepcopy(text_embeddings.detach()),
            'caption': [c[0] for c in batch['captions']],
            'path': batch['fname']
        }

        self.validation_outputs.append(args)

    def on_validation_epoch_end(self, prefix='val'):
        outputs = self.validation_outputs

        # concatenate metadata
        paths = np.array([p for b in outputs for p in b['path']])
        captions = np.array([p for b in outputs for p in b['caption']])

        # audios in clotho can have five captions
        # this snippet discards every occurrence of a duplicate audio
        #
        target = [] # prediction targets for later
        select = [] # indices of the first occurrence for later
        first_occurrence = {} # temporary cache to keep track of first occurrences
        for i, p in enumerate(paths): # iterate over all paths
            index = first_occurrence.get(p)
            if index is None:  # First time seeing this path
                index = len(first_occurrence)
                first_occurrence[p] = index
                select.append(i) # these audios will be selected
            target.append(index) # all paths need a target - choose the correct one
        paths = paths[select]

        # concatenate embeddings
        audio_embeddings = torch.cat([o['audio_embeddings'] for o in outputs])[select]# only select unique audios
        text_embeddings = torch.cat([o['text_embeddings'] for o in outputs])

        # concatenate global ranking
        C_text_to_audio = torch.matmul(text_embeddings, audio_embeddings.T)
        C_audio_to_text = C_text_to_audio.T  # Transpose for audio-to-text retrieval

        # ========== Text-to-Audio Retrieval Metrics ==========
        # get top 10 for text-to-audio
        top_ten_t2a = C_text_to_audio.topk(10, dim=1)[1].detach().cpu().numpy()
        target = np.array(target)

        # recall metrics for text-to-audio
        r_1_t2a = (top_ten_t2a[:, :1] == target[:, None]).sum(axis=1).mean()
        r_5_t2a = (top_ten_t2a[:, :5] == target[:, None]).sum(axis=1).mean()
        r_10_t2a = (top_ten_t2a == target[:, None]).sum(axis=1).mean()

        # mAP@10 for text-to-audio
        AP_t2a = 1 / ((top_ten_t2a == target[:, None]).argmax(axis=1) + 1)
        AP_t2a[~(top_ten_t2a == target[:, None]).any(axis=1)] = 0
        mAP_t2a = AP_t2a.mean()

        # log text-to-audio retrieval performance
        prefix = 'text-to-audio/' + prefix
        self.log(f'{prefix}/R@1', r_1_t2a)
        self.log(f'{prefix}/R@5', r_5_t2a)
        self.log(f'{prefix}/R@10', r_10_t2a)
        self.log(f'{prefix}/mAP@10', mAP_t2a)

        # ========== Audio-to-Text Retrieval Metrics ==========
        # For audio-to-text, we need to find which captions correspond to each audio
        # Create a mapping: audio_idx -> list of caption indices that match this audio
        audio_to_caption_indices = {}
        for caption_idx, audio_idx in enumerate(target):
            if audio_idx not in audio_to_caption_indices:
                audio_to_caption_indices[audio_idx] = []
            audio_to_caption_indices[audio_idx].append(caption_idx)

        # get top 10 for audio-to-text
        top_ten_a2t = C_audio_to_text.topk(10, dim=1)[1].detach().cpu().numpy()
        
        # Calculate recall metrics for audio-to-text
        # For each audio, check if any of its ground truth captions are in top-k
        r_1_a2t_list = []
        r_5_a2t_list = []
        r_10_a2t_list = []
        AP_a2t_list = []
        
        for audio_idx in range(len(paths)):
            true_caption_indices = set(audio_to_caption_indices.get(audio_idx, []))
            retrieved_indices = top_ten_a2t[audio_idx]
            
            # R@1
            r_1_a2t_list.append(1 if retrieved_indices[0] in true_caption_indices else 0)
            
            # R@5
            r_5_a2t_list.append(1 if any(idx in true_caption_indices for idx in retrieved_indices[:5]) else 0)
            
            # R@10
            r_10_a2t_list.append(1 if any(idx in true_caption_indices for idx in retrieved_indices[:10]) else 0)
            
            # mAP@10
            # Find the rank of the first relevant caption
            relevant_ranks = [rank + 1 for rank, idx in enumerate(retrieved_indices) if idx in true_caption_indices]
            if relevant_ranks:
                AP_a2t_list.append(1.0 / relevant_ranks[0])
            else:
                AP_a2t_list.append(0.0)
        
        r_1_a2t = np.mean(r_1_a2t_list)
        r_5_a2t = np.mean(r_5_a2t_list)
        r_10_a2t = np.mean(r_10_a2t_list)
        mAP_a2t = np.mean(AP_a2t_list)

        # log audio-to-text retrieval performance
        prefix = 'audio-to-text/' + prefix
        self.log(f'{prefix}_a2t/R@1', r_1_a2t)
        self.log(f'{prefix}_a2t/R@5', r_5_a2t)
        self.log(f'{prefix}_a2t/R@10', r_10_a2t)
        self.log(f'{prefix}_a2t/mAP@10', mAP_a2t)

        if os.path.exists(f'resources/metadata_eval.csv') and prefix == 'test':

            matched_files = pd.read_csv(f'resources/metadata_eval.csv')
            matched_files["audio_filenames"] = matched_files["audio_filenames"].transform(lambda x: ast.literal_eval(x))

            def get_ranks(c, r):
                ranks = [i.item() for i in torch.argsort(torch.argsort(-c))[r]]
                return ranks

            # Create mapping dictionaries for safe lookup
            captions_list = captions.tolist()
            captions_to_index = {cap: idx for idx, cap in enumerate(captions_list)}
            paths_list = paths.tolist()
            paths_to_index = {path: idx for idx, path in enumerate(paths_list)}

            # index of query in C - use safe lookup with fallback
            def safe_caption_index(x):
                if x in captions_to_index:
                    return captions_to_index[x]
                else:
                    # Try case-insensitive and stripped version
                    x_normalized = x.lower().strip() if isinstance(x, str) else str(x).lower().strip()
                    for cap, idx in captions_to_index.items():
                        if isinstance(cap, str) and cap.lower().strip() == x_normalized:
                            return idx
                    # If still not found, return None (will be filtered out)
                    print(f"Warning: Query '{x}' not found in captions. Skipping this entry.")
                    return None

            matched_files["query_index"] = matched_files["query"].transform(safe_caption_index)
            
            # Filter out rows where query_index is None
            matched_files = matched_files[matched_files["query_index"].notna()].copy()
            
            if len(matched_files) == 0:
                print("Warning: No valid queries found in metadata_eval.csv after matching. Skipping multiple positives mAP calculation.")
            else:
                matched_files["query_index"] = matched_files["query_index"].astype(int)

                # new ground truth - use safe lookup
                def safe_path_indices(x):
                    indices = []
                    for y in x:
                        if y in paths_to_index:
                            indices.append(paths_to_index[y])
                        else:
                            # Try to find similar path (case-insensitive)
                            y_normalized = y.lower().strip() if isinstance(y, str) else str(y).lower().strip()
                            found = False
                            for path, idx in paths_to_index.items():
                                if isinstance(path, str) and path.lower().strip() == y_normalized:
                                    indices.append(idx)
                                    found = True
                                    break
                            if not found:
                                print(f"Warning: Audio filename '{y}' not found in paths. Skipping this filename.")
                    return indices

                matched_files["new_audio_indices"] = matched_files["audio_filenames"].transform(safe_path_indices)
                
                # Filter out rows where new_audio_indices is empty
                matched_files = matched_files[matched_files["new_audio_indices"].apply(lambda x: len(x) > 0)].copy()
                
                if len(matched_files) > 0:
                    matched_files["TP_ranks"] = matched_files.apply(lambda row: get_ranks(C_text_to_audio[row["query_index"]], row["new_audio_indices"]), axis=1)

                    def average_precision_at_k(relevant_ranks, k=10):
                        relevant_ranks = sorted(relevant_ranks)
                        ap = 0.0
                        for i, rank in enumerate(relevant_ranks, start=1):
                            if rank >= k:
                                break
                            ap += i / (rank + 1) # precision at threshold
                        return ap / len(relevant_ranks)  # Normalize by total number of relevant items

                    new_mAP = matched_files["TP_ranks"].apply(lambda ranks: average_precision_at_k(ranks, 10)).mean()
                    self.log(f'{prefix}_multiple_positives/mAP@10', new_mAP)
        # empty cached batches from validation loop
        self.validation_outputs.clear()


    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end(prefix='test')

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False
        )

        return optimizer

    def lr_scheduler_step(self, batch_idx):

        steps_per_epoch = self.trainer.num_training_batches

        min_lr = self.kwargs['min_lr']
        max_lr = self.kwargs['max_lr']
        current_step = self.current_epoch * steps_per_epoch + batch_idx
        warmup_steps = self.kwargs['warmup_epochs'] * steps_per_epoch
        total_steps = (self.kwargs['warmup_epochs'] + self.kwargs['rampdown_epochs']) * steps_per_epoch
        decay_steps = total_steps - warmup_steps

        if current_step < warmup_steps:
            # Linear warmup
            lr = min_lr + (max_lr - min_lr) * (current_step / warmup_steps)
        elif current_step < total_steps:
            # Cosine decay
            decay_progress = (current_step - warmup_steps) / decay_steps
            lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
        else:
            # Constant learning rate
            lr = min_lr

        for param_group in self.optimizers(use_pl_optimizer=False).param_groups:
            param_group['lr'] = lr

        self.log('train/lr', lr, sync_dist=True)

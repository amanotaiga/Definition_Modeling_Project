import argparse
import glob
import logging
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import random
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from lightning_base import BaseTransformer, add_generic_args, generic_train
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup, AdamW
from pytorch_lightning.callbacks import EarlyStopping
from torch.nn import CrossEntropyLoss
from nltk.translate import bleu_score

try:
    from transformers import MT5ForConditionalGeneration
    from .utils import (
        assert_all_frozen,
        lmap,
        flatten_list,
        pickle_save,
#         save_git_info,
        save_json,
        freeze_params,
        calculate_rouge,
#         get_git_info,
        ROUGE_KEYS,
        Seq2SeqDataset,
    )

    from .callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
except ImportError:
    from utils import (
        Seq2SeqDataset,
        assert_all_frozen,
        lmap,
        flatten_list,
        pickle_save,
#         save_git_info,
        save_json,
        freeze_params,
        calculate_rouge,
#         get_git_info,
        ROUGE_KEYS,
    )
    from callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42) 
logger = logging.getLogger(__name__)

class SummarizationModule(pl.LightningModule):
    loss_names = ["loss"]
    metric_names = ["bleu"]
    val_metric = "bleu"

    def __init__(self, hparams, **kwargs):
        super(SummarizationModule, self).__init__()
        self.hparams = hparams
        self.output_dir = self.hparams.output_dir
        if 'mt5' in hparams.model_name_or_path:
            pass
#             self.model = MT5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
        
#         save_git_info(self.hparams.output_dir)
        self.metrics_save_path = Path(self.hparams.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.hparams.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
        )
        self.option = self.hparams.option
        self.resume_trn = self.hparams.resume_trn
    
        self.sample = self.hparams.sample
        self.beams_penalty = self.hparams.beams_penalty
        self.beams_group = self.hparams.beams_group
        self.num_beams = self.hparams.num_beams
        self.type_path = ''
        
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
            "test_val": self.hparams.n_test,            
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
            "test_val": self.hparams.test_max_target_length,            
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"

        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

#         self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.decoder_start_token_id = None
        self.dataset_class = Seq2SeqDataset

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)
 
    
    def _step(self, batch: dict) -> Tuple:

        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        
        decoder_input_ids = self.model._shift_right(target_ids)
        lm_labels = target_ids
        outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=decoder_input_ids, use_cache=False)

        # Same behavior as modeling_bart.py
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        lm_logits = outputs[0]
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1))
        
        return (loss,)

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs) -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
#         rouges = {k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]}
        rouges = {k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names}
        rouge_tensor: torch.FloatTensor = torch.tensor(rouges[self.val_metric]).type_as(loss)
        rouges.update({k: v.item() for k, v in losses.items()})
        losses.update(rouges)
        metrics = {f"val_avg_{k}": x for k, x in losses.items()}
        metrics["step_count"] = self.step_count
        self.save_metrics(metrics, 'val')  # writes to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])
 
        return {"log": metrics, "preds": preds, "val_loss": loss, f"val_{self.val_metric}": rouge_tensor}

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)

    def calculate_bleu(self, preds, targets, fast):  
        if fast:
            return {'bleu':self.step_count}
        avg = []
        for item in zip(preds, targets):
            pred = item[0].split() if len(item[0].split())>0 else ['dummy']
            tar = item[1].split() if len(item[1].split())>0 else ['dummy']
            score = bleu_score.sentence_bleu([tar], pred,
                                     smoothing_function=bleu_score.SmoothingFunction().method2,auto_reweigh=True)
            avg.append(score)
        return {'bleu':sum(avg)/(len(avg)+1)}
        
    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()
        fast = False
        generated_ids = batch["decoder_input_ids"]
        if not fast:
            generated_ids = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
            )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["decoder_input_ids"])
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
#         rouge: Dict = self.calc_generative_metrics(preds, target)
        rouge: Dict = self.calculate_bleu(preds, target, fast)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(preds=preds, target=target, **rouge)
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **rouge)            
        return base_metrics

    def test_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        bad_word_list = batch["target_word"]
#         loss = self._step(batch)
        loss = torch.tensor(0.0)
        losses = []
        log_list = []
        avg_list = []
        sum_list = []        
        if (not args.option.startswith('t5_specific') and not args.option.startswith('forward') and not args.option.startswith('t5_general')):
            generated_ids = self.model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                num_beams=self.num_beams,
                repetition_penalty=1.0,
                length_penalty=0.8,        
                no_repeat_ngram_size = 1,
                early_stopping=True,
                use_cache=True,
                bad_words_ids = bad_word_list,
                num_return_sequences=self.num_beams  
            )

            preds = [
                self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in generated_ids
            ]
            targets = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
        else:
            source_ids, source_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]            
            decoder_input_ids = self.model._shift_right(target_ids)
            outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
            loss_fct = CrossEntropyLoss(ignore_index=pad_token_id)
            labels_hat = torch.argmax(outputs[0], dim=2)
            preds = [
                self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in labels_hat
            ]
            targets = [
                self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in target_ids
            ]       
            m = torch.nn.LogSoftmax(dim=2) 
            after_logsoftmax = m(outputs[0])         
            for i in range(len(outputs[0])):
                # calculate cross entropy by sentence not by batch
                loss = loss_fct(outputs[0][i].view(-1, outputs[0][i].size(-1)), target_ids[i].cuda().view(-1))

                for j in range(len(after_logsoftmax[i])):
                    log_list.append(after_logsoftmax[i][j][target_ids[i][j].item()].item())
                    if target_ids[i][j].item()==1:
                        break
                avg_list.append(str(sum(log_list)/(len(log_list)+0.001)))
                sum_list.append(str(sum(log_list)))
                log_list = []                    
                losses.append(str(loss.item()))

        return {"test_loss": loss, "preds": preds, "targets": targets, "losses": losses, "log_sum": sum_list, "log_avg": avg_list}

    def test_end(self, outputs):
        output_test_predictions_file = os.path.join(
            args.output_dir, "{}_predictions.txt".format(self.hparams.test_dataset.split('_')[-1]))
        output_test_targets_file = os.path.join(
            args.output_dir, "{}_targets.txt".format(self.hparams.test_dataset.split('_')[-1]))
        output_test_losses_file = os.path.join(
            args.output_dir, "{}_losses.txt".format(self.hparams.test_dataset.split('_')[-1]))

        # write predictions and targets
        if self.option=="":
            # output predictions
            with open(output_test_predictions_file, "w",encoding='utf-8') as p_writer:
                for output_batch in outputs:
                    p_writer.writelines(s + "\n" for s in output_batch["preds"])
                p_writer.close()

        else:
        # output loss(scores)            
            with open(output_test_losses_file, "w",encoding='utf-8') as r_writer:
                for output_batch in outputs:
                    r_writer.writelines(s + "\n" for s in output_batch["losses"])
                r_writer.close()

#         # output targets
#         with open(output_test_targets_file, "w",encoding='utf-8') as t_writer:
#             for output_batch in outputs:            
#                 t_writer.writelines(s + "\n" for s in output_batch["targets"])
#             t_writer.close()
     
        return self.test_epoch_end(outputs)
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()

        return {"avg_test_loss": avg_loss}

    def get_dataset(self, type_path, option, self_ref) -> Seq2SeqDataset:
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            max_target_length=max_target_length,
            option = option,
            self_ref = self_ref,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path, self.hparams.option, self.hparams.self_ref)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            sampler=None,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.gpus)))
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        if max(scheduler.get_last_lr()) > 0:
            warnings.warn("All learning rates are 0")
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        self.type_path = self.hparams.test_dataset
        return self.get_dataloader(self.type_path, batch_size=self.hparams.eval_batch_size)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)

        parser.add_argument(
            "--max_source_length",
            default=200,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=150,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=150,  # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=150,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )     
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--logger_name", type=str, choices=["default"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        return parser
    
    
def main(args, model=None) -> SummarizationModule:
    
    Path(args.output_dir).mkdir(exist_ok=True)
    model: SummarizationModule = SummarizationModule(args) 
            
    dataset = Path(args.data_dir).name
    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        logger = True  # don't pollute wandb logs unnecessarily
        
    ck = False
    if args.resume_trn:
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
        ck = checkpoints[-1]
         
    es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)

    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(args.output_dir, model.val_metric),
        early_stopping_callback=es_callback,
        resume_from_checkpoint = ck,
        logger=logger,
    )        
    pickle_save(model.hparams, args.output_dir+"hparams.pkl")
    
    if not args.do_predict:
        return model
    print(args.do_predict)
    model.hparams.test_checkpoint = ""
    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]
    trainer.logger.log_hyperparams(model.hparams)

    # test() without a model tests using the best checkpoint automatically
    trainer.test(model)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)

    parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())

    parser.add_argument(
            "--option",
            type=str,
            default = '',
            help="t5_general or t5_specific",
        )
    parser.add_argument(
            "--self_ref",
            action="store_false",
            default = True,
            help="containing self-reference or not",
        )      
    parser.add_argument(
            "--resume_trn",
            action="store_true",
            default = False,
            help="resume training",
        )  
    parser.add_argument(
            "--test_dataset",
            type=str,
            default = 'test',
            help="generate prediction for test set or validation set ",
        )   
    parser.add_argument(
            "--sample",
            type=int,
            default = 100,
            help="how many samples are used",
        )       
    parser.add_argument(
            "--beams_penalty",
            type=float,
            default = 1.0,
            help="penalty for diverse beam search",
        ) 
    parser.add_argument(
            "--beams_group",
            type=int,
            default = 1,
            help="how many group of beams",
        )    
    parser.add_argument(
            "--num_beams",
            type=int,
            default = 100,
            help="The number of beam search",
        )      
    args = parser.parse_args()

    main(args)

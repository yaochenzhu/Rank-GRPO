import os
import re
import json
import regex
import pickle

import datetime
from time import sleep

import hashlib
import numpy as np

from collections import defaultdict
from editdistance import eval as distance
from transformers import TrainerCallback

def del_parentheses(text):
    pattern = r"\([^()]*\)"
    return re.sub(pattern, "", text.strip())

def del_space(text):
    pattern = r"\s+"
    return re.sub(pattern, " ", text.strip()).strip()

def del_numbering(text):
    pattern = r"^\d+\s*[\.\)、\-\—\–]\s*"
    return re.sub(pattern, "", text.strip()).strip()

def del_format(text):
    text = text.strip()
    # Repeatedly remove all *, _, -, # from both ends (to handle nesting)
    while True:
        new_text = re.sub(r'^([\*\_\-\#]+)', '', text).strip()
        new_text = re.sub(r'([\*\_\-\#]+)$', '', new_text).strip()

        new_text = re.sub(
            r'^(?:[\*\_\-\#]+\s+|(?:\d+\s*[\.\)、\-\—\–]\s+))', '', new_text
        ).strip()
    
        # Remove leading '#' with or without spaces after it
        new_text = re.sub(r'^#+\s*', '', new_text)

        if new_text == text:
            break
        text = new_text
    return text

def remove_quotes(s):
    if (s.startswith('"') and s.endswith('"')) \
        or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

def process_rec_raw(item, raw_rec_field, rec_field):
    rec_list_raw = item[raw_rec_field]
    rec_list_raw = re.sub(r'\n+', '\n', rec_list_raw)
    
    # Split by newline and strip each line to remove leading/trailing whitespace
    lines = [line.strip() for line in rec_list_raw.strip().split('\n') if line.strip()]
        
    try:
        pattern = r"(.+?)\s+\((\d{4})\)"
        
        rec_list = []
        for line in lines:
            line = remove_quotes(del_format(del_space(line)))
            match = re.match(pattern, line)
            if match:
                movie_name = match.group(1)
                new_movie_name = remove_quotes(del_format(del_space(del_parentheses(movie_name.strip()))))
                while new_movie_name != movie_name:
                    movie_name = new_movie_name
                    new_movie_name = remove_quotes(del_format(del_space(del_parentheses(movie_name.strip()))))
                    
                year = int(match.group(2))
                rec_list.append((movie_name, year))
        item[rec_field] = rec_list
        error = False
    except:
        item[rec_field] = []
        print("error")
        error = True
    return error, item


class StepLRSchedulerCallback(TrainerCallback):
    """
    Step-based learning rate scheduler for Hugging Face Trainer.

    Args:
        schedule (list[tuple[int, float]]):
            List of (step_threshold, lr_value) pairs, e.g.
            [(1000, 5e-7), (4000, 1e-7), (8000, 5e-8), (12000, 1e-8)]
        verbose (bool): Whether to print LR changes.
    """
    def __init__(self, schedule=None, verbose=True):
        super().__init__()
        # Default schedule if none provided
        self.schedule = schedule
        self.verbose = verbose

    def on_step_begin(self, args, state, control, **kwargs):
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return control
        step = state.global_step

        for threshold, lr in reversed(self.schedule):
            if step >= threshold:
                for group in optimizer.param_groups:
                    group["lr"] = lr
                if self.verbose and getattr(args, "local_rank", -1) in [-1, 0]:
                    print(f"🔽 LR dropped to {lr:.1e} at step {step}")
                break
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        optimizer = kwargs.get("optimizer")
        if logs and "learning_rate" in logs and optimizer is not None:
            logs["learning_rate"] = optimizer.param_groups[0]["lr"]
        return control


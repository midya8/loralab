"""Frozen original LoRALab epoch boundaries, without changing online training.

All four original widths and 31 states per width are retained. An epoch boundary
is not every SGD update: each original epoch has 120 sequential examples.
"""
from __future__ import annotations

import copy
import math
import random

from .eval import build_fixture, run_evaluation
from .lora import LoRAClassifier
from .coordinates import audit, norm


def _metrics(adapter, examples):
    loss = 0.0
    correct = 0
    for x, y in examples:
        z = adapter.logits(x)
        top = max(z)
        loss += top-z[y]+math.log(math.fsum(math.exp(v-top) for v in z))
        correct += int(max(range(len(z)), key=lambda j: (z[j], -j)) == y)
    return {"mean_cross_entropy": loss/len(examples), "accuracy": correct/len(examples)}


def build_archive():
    fx = build_fixture()
    base = fx['base']
    datasets = {key: [{"x": x[:], "y": y} for x,y in fx[key]]
                for key in ['trainA','testA','trainB','testB']}
    runs = []
    for rank in [1,2,4,8]:
        adapter = LoRAClassifier(base,rank,random.Random(0))
        states=[]
        for epoch in range(31):
            states.append({"epoch": epoch, "completed_online_updates": epoch*len(fx['trainB']),
                           "A": copy.deepcopy(adapter.A), "B": copy.deepcopy(adapter.B),
                           "A_norm": norm(adapter.A), "B_norm": norm(adapter.B),
                           "trainB": _metrics(adapter,fx['trainB']),
                           "testA": _metrics(adapter,fx['testA']),
                           "testB": _metrics(adapter,fx['testB'])})
            if epoch < 30:
                adapter.train_epoch(fx['trainB'],fx['lr'])
        runs.append({"factor_width": rank, "factor_parameters": rank*(base.dim+base.n_classes),
                     "matrix_rank_upper_bound": min(rank,base.dim,base.n_classes), "states": states})
    return {"schema": "loralab.original-epochs.v1", "original_report": run_evaluation(fx),
            "W": copy.deepcopy(base.W), "bias": base.b[:], "datasets": datasets,
            "online_learning_rate": fx['lr'], "runs": runs,
            "recording": "All 31 epoch boundaries per original width; not all 3600 online updates."}


def problem_at(archive, rank=2, epoch=30, batch='first_train'):
    if isinstance(rank,bool) or not isinstance(rank,int) or rank not in [1,2,4,8]:
        raise ValueError('factor width must be one of 1, 2, 4, 8')
    if isinstance(epoch,bool) or not isinstance(epoch,int) or not 0<=epoch<=30:
        raise ValueError('epoch must be an integer from 0 through 30')
    if batch not in ['first_train','full_train']:
        raise ValueError('batch must be first_train or full_train')
    state=next(run for run in archive['runs'] if run['factor_width']==rank)['states'][epoch]
    examples=archive['datasets']['trainB']
    if batch=='first_train': examples=examples[:1]
    return {"W": archive['W'], "bias": archive['bias'], "A": state['A'], "B": state['B'], "examples": examples}


def trajectory_report(archive, rank=2, epoch=30, batch='first_train', scale=4, learning_rate=.1):
    result=audit(problem_at(archive,rank,epoch,batch),scalar_gauge=scale,learning_rate=learning_rate)
    result['fixture_selection']={"factor_width":rank,"epoch":epoch,"batch":batch,
                                 "original_online_learning_rate":archive['online_learning_rate'],
                                 "diagnostic_is_retraining":False}
    return result

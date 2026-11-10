"""Compare pretrained Chronos-Bolt with saved Week 8 model on identical forecast origins.

Optional dependency: pip install -r week8/requirements_chronos.txt
First run downloads the model weights from Hugging Face. This script does not
train Chronos or use any 2024 observations for training.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from forecast_core import (HORIZONS, make_windows, read_data, sample_windows,
                           baseline, forecast_ml)
from train_evaluate import append_predictions, metric_rows

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path,default=ROOT/'data/processed/Dhaka_CAMS_Cleaned_2012_2024.csv')
    ap.add_argument('--output', type=Path,default=ROOT/'week8/outputs')
    ap.add_argument('--model-id', default='amazon/chronos-bolt-tiny')
    ap.add_argument('--origins-per-station',type=int,default=12,
                    help='Same sampled dates for every method; set 0 for full test (slower)')
    ap.add_argument('--horizons', type=int, nargs='+', default=[24,168],choices=[24,168])
    args=ap.parse_args()
    try:
        import torch
        from chronos import BaseChronosPipeline
    except ImportError as exc:
        ap.error('Chronos is not installed. Run: pip install -r week8/requirements_chronos.txt')
    data=read_data(args.data)
    args.output.mkdir(parents=True,exist_ok=True)
    print('Loading Chronos model...',flush=True)
    pipeline=BaseChronosPipeline.from_pretrained(args.model_id, device_map='cpu')
    all_rows=[]
    for horizon in args.horizons:
        model_file=ROOT/'week8/models'/f'pm25_{horizon}h_gradient_boosting.joblib'
        if not model_file.exists():
            ap.error(f'First run: python week8/src/train_evaluate.py (missing {model_file.name})')
        ml=joblib.load(model_file)['model']
        windows,_=make_windows(data,horizon)
        chosen = (sample_windows(windows,'validation',args.origins_per_station)+
                  sample_windows(windows,'test',args.origins_per_station))
        print(f'{horizon}h: {len(chosen)} windows',flush=True)
        for idx, w in enumerate(chosen):
            context=torch.tensor(w.history,dtype=torch.float32)
            pred=pipeline.predict(context,prediction_length=horizon)
            # Chronos-Bolt: tensor [batch, quantile, forecast_hour].
            # 0.5 quantile is the median prediction, index 4 in official Bolt models.
            quantiles=getattr(pipeline,'quantiles',[.1,.2,.3,.4,.5,.6,.7,.8,.9])
            median_index=list(quantiles).index(.5)
            pred_chronos = pred[0,median_index,:].detach().cpu().numpy()
            forecasts={'Chronos-Bolt zero-shot':np.maximum(0,pred_chronos),
                       'Gradient boosting':forecast_ml(ml,w.history,w.station,w.origin,horizon),
                       'Daily repeat':baseline(w.history,horizon,'Daily repeat'),
                       'Persistence':baseline(w.history,horizon,'Persistence')}
            for name, forecast in forecasts.items():
                append_predictions(all_rows,w,name,forecast,horizon)
            if (idx+1)%10==0:print(f'  {idx+1}/{len(chosen)}',flush=True)
    points=pd.DataFrame(all_rows)
    points.to_csv(args.output/'chronos_comparison_points.csv',index=False)
    summary=pd.DataFrame(metric_rows(points))
    summary.to_csv(args.output/'chronos_comparison.csv',index=False)
    print(summary.to_string(index=False))

if __name__=='__main__':main()

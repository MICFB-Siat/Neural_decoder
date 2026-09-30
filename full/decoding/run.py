import argparse
import csv
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description='Saved-weight inference; external data required.')
    p.add_argument('panel', choices=['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'J'])
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--dataset', choices=['StudyForrest','Alice','LPPCHK','LPPC_EN'], default='StudyForrest')
    p.add_argument('--unclip', type=Path)
    p.add_argument('--indices', nargs='+', type=int)
    p.add_argument('--features-only', action='store_true')
    p.add_argument('--evaluate-images', action='store_true')
    p.add_argument('--llm', type=Path)
    p.add_argument('--bge', type=Path)
    p.add_argument('--subject', action='append')
    p.add_argument('--device', default='cpu')
    p.add_argument('--limit', type=int)
    p.add_argument('--output', type=Path)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        p.error('--limit must be positive')
    out = a.output or ROOT / 'output' / a.panel
    if a.panel in ['A', 'B', 'C']:
        if not a.llm and not a.check_only:
            p.error('--llm is required for language inference')
        sys.path.insert(0, str(ROOT / 'language'))
        import torch
        torch.set_num_threads(4)
        from generation_runtime import run_generation
        lang, tokens, chars = {'A': ('DE' if a.dataset=='StudyForrest' else ('CN' if a.dataset=='LPPCHK' else 'EN'), 80, 200), 'B': ('CN', 96, 50), 'C': ('FR', 96, 160)}[a.panel]
        run_generation(weight_dir=ROOT / a.panel / 'weights' / (a.dataset if a.panel == 'A' else 'model'),
                       data_dir=a.data, output_dir=out, lang=lang,
                       device_name=a.device, batch_size=16, beam=3, max_new=tokens,
                       polish_max_chars=chars, subjects=a.subject,
                       llm_path=a.llm or Path('frozen_models/Phi-4-mini-instruct'),
                       check_only=a.check_only, limit=a.limit)
        if a.bge and not a.check_only:
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            from sentence_transformers import SentenceTransformer
            rows = list(csv.DictReader((out / 'predictions.csv').open(encoding='utf-8-sig')))
            model = SentenceTransformer(str(a.bge), device=a.device)
            clean = lambda s: ''.join(s.split()) if lang == 'CN' else s.strip()
            ref = model.encode([clean(r['ground_truth']) or ' ' for r in rows], normalize_embeddings=True)
            pred = model.encode([r['output'] or ' ' for r in rows], normalize_embeddings=True)
            for row, score in zip(rows, (ref * pred).sum(-1)):
                row['BGEScore'] = float(score)
            with (out / 'scores.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            from plot_results import language_scores
            language_scores(rows,out,a.panel)
        return
    if a.check_only:
        p.error('--check-only is available for A/B/C')
    if a.panel in ['G','H','J'] and not a.features_only:
        if not a.unclip: p.error('--unclip is required for image reconstruction')
        if (a.panel=='H' or a.evaluate_images) and a.limit is not None and a.limit<2:
            p.error('Image identification metrics require --limit >= 2')
        from reconstruction import reconstruct
        import numpy as np
        import json
        for subject in a.subject or ['sub-01']:
            ids, images = reconstruct(a.data/subject, ROOT/'G/weights'/subject, out/subject,
                                      a.device,a.unclip,a.limit,a.indices)
            if a.evaluate_images or a.panel=='H':
                import torch
                from image_metrics import evaluate_pair
                real=np.load(a.data/subject/'images_test.npy',mmap_mode='r')[ids]
                if real.ndim!=4: raise ValueError('images_test.npy must be N x 3 x H x W')
                scores=evaluate_pair(torch.as_tensor(real).float(),torch.as_tensor(images).float(),torch.device(a.device))
                (out/subject/'image_metrics.json').write_text(json.dumps(scores,indent=2))
                from plot_results import image_scores
                image_scores(scores,out/subject,a.panel)
        return
    from metrics import evaluate
    evaluate(a.panel, a.data, ROOT, out, a.subject, a.device, a.limit)


if __name__ == '__main__':
    main()

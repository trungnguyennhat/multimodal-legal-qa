"""Detect traffic signs in corpus images with local YOLOE and preserve citation provenance."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

from hybrid_rag.full_image_method.corpus_image_embedding import (
    CORPUS, MODEL_DIR, Progress, check_hashes, corpus_inputs, file_hash, hashes_of,
    read_lines, save, verified_model, write_lines,
)
from hybrid_rag.yolo_e_train.yoloe_preview import ASSETS, MODELS, VERSION

BASE = ROOT / 'artifacts/experiments/hybrid_rag/crop_image_method'
SCHEMA = 1
PROMPTS = [
    'traffic sign', 'circular traffic sign', 'red circular prohibitory traffic sign',
    'blue circular mandatory traffic sign', 'triangular warning traffic sign',
    'yellow triangular traffic sign with red border', 'inverted triangular traffic sign',
    'rectangular traffic sign', 'square traffic sign', 'blue rectangular information traffic sign',
    'green direction traffic sign', 'arrow-shaped direction traffic sign',
    'white supplementary traffic sign', 'diamond-shaped priority road sign', 'red octagonal stop sign',
]
ALPHA_RGB = (255, 255, 255)
STEPS = {'corpus_cropping', 'query_cropping_dev', 'corpus_embedding', 'query_embedding_dev',
         'query_retrieval_corpus', 'query_retrieval_example', 'query_retrieval_hybrid', 'image_text_retrieval'}
SHARED_CODE = [ROOT / ('src/' + p) for p in (
    'hybrid_rag/full_image_method/corpus_image_embedding.py',
    'hybrid_rag/yolo_e_train/yoloe_preview.py',
    'text_based_rag/corpus_embedding.py', 'text_based_rag/corpus_image_captioning.py',
)]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def run_step(args, name, action, entrypoint, dependencies=()):
    folder = BASE / name
    if (name not in STEPS or folder.is_symlink() or folder.resolve() != folder.absolute() or folder.resolve() == BASE.resolve()
            or not folder.resolve().is_relative_to(BASE.resolve()) or not BASE.resolve().is_relative_to(ROOT)):
        raise ValueError('Output phải nằm trong đúng bước của crop_image_method, không dùng symlink.')
    if folder.exists():
        if not args.overwrite:
            raise FileExistsError(f'{folder} đã tồn tại; dùng --overwrite cho đúng bước này.')
        # Remove only the verified step directory so old crops cannot survive a configuration change.
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    code = {Path(__file__).resolve(), entrypoint.resolve(), *SHARED_CODE, *dependencies}
    meta = {'status': 'RUNNING', 'schema_version': SCHEMA, 'representation': 'crop_image_method',
            'step': name, 'created_at': datetime.now(timezone.utc).isoformat(), 'arguments': vars(args),
            'code_sha256': hashes_of(sorted(code)), 'input_sha256': {},
            'packages': {n: version(n) for n in ('torch', 'transformers', 'numpy', 'pillow', 'ultralytics', 'clip')}}
    save(folder / 'manifest.json', meta)
    progress = Progress()
    progress.worker.start()
    try:
        action(args, folder, meta, progress)
        for key in ('code_sha256', 'input_sha256', 'output_sha256'):
            check_hashes(meta[key])
        meta.update(status='COMPLETED', elapsed_seconds=time.monotonic() - progress.started)
        save(folder / 'manifest.json', meta)
        print(f'[output] {folder}', flush=True)
    except BaseException as error:
        meta.update(status='FAILED', error=f'{type(error).__name__}: {error}', failed_stage=progress.stage,
                    processed=progress.done, total=progress.total, elapsed_seconds=time.monotonic() - progress.started)
        save(folder / 'manifest.json', meta)
        raise
    finally:
        progress.stop.set()
        progress.worker.join()
        progress.log()


def load_artifact(folder, names):
    folder = folder.resolve()
    if not folder.is_relative_to(ROOT):
        raise ValueError('Artifact nguồn phải nằm trong project.')
    manifest = folder / 'manifest.json'
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    required = {str((folder / n).relative_to(ROOT)) for n in names}
    if (meta.get('status') != 'COMPLETED' or not meta.get('code_sha256')
            or not meta.get('input_sha256') or not required <= set(meta.get('output_sha256', {}))):
        raise ValueError(f'Artifact chưa COMPLETED hoặc thiếu fingerprint/output: {folder}')
    for key in ('code_sha256', 'input_sha256', 'output_sha256'):
        check_hashes(meta[key])
    hashes = {**meta['input_sha256'], **meta['code_sha256'], **meta['output_sha256'], **hashes_of([manifest])}
    return meta, hashes


def load_step(name, names):
    folder = BASE / name
    meta, hashes = load_artifact(folder, names)
    if (meta.get('schema_version') != SCHEMA or meta.get('representation') != 'crop_image_method'
            or meta.get('step') != name):
        raise ValueError(f'Schema crop không hợp lệ: {folder}')
    return meta, folder, hashes


def corpus_inventory(progress):
    images, registry, hashes = corpus_inputs(progress)
    if len(images) != 787:
        raise ValueError('Inventory corpus khác 787 ảnh đã duyệt; kiểm tra dữ liệu trước khi chạy.')
    return [{k: v for k, v in r.items() if k != 'vector_row'} for r in images], registry, hashes


def detector_assets():
    manifest = ROOT / 'temp/sign_preview/yolo_e_train/download_manifest.json'
    assets = json.loads(manifest.read_text(encoding='utf-8'))
    if version('ultralytics') != VERSION:
        raise RuntimeError(f'Cần ultralytics {VERSION} trong .venv; không tự nâng/hạ package.')
    paths = [manifest]
    for name, size in ASSETS.items():
        path = MODELS / name
        if (not path.is_file() or path.stat().st_size != size or
                file_hash(path) != assets.get(name, {}).get('sha256')):
            raise ValueError(f'Model local thiếu hoặc sai hash: {path}; không tự tải lại.')
        paths.append(path)
    return hashes_of(paths)


def rgb_image(path):
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGBA')
        return Image.alpha_composite(Image.new('RGBA', image.size, (*ALPHA_RGB, 255)), image).convert('RGB')


def pixel_hash(image):
    digest = hashlib.sha256(f'RGB:{image.width}:{image.height}:'.encode('ascii'))
    digest.update(image.tobytes())
    return digest.hexdigest()


def detect_images(args, folder, meta, progress, images, queries, hashes):
    progress.set('verify local YOLOE, MobileCLIP and DINOv2')
    hashes.update(detector_assets())
    dino = verified_model()
    hashes.update(dino['output_sha256'])
    hashes.update(hashes_of([MODEL_DIR / 'manifest.json']))
    config = {'model': 'yoloe-v8l-seg.pt', 'weights_sha256': file_hash(MODELS / 'yoloe-v8l-seg.pt'),
              'text_encoder_sha256': file_hash(MODELS / 'mobileclip_blt.ts'), 'ultralytics': VERSION,
              'prompts': PROMPTS, 'conf': args.conf, 'imgsz': args.imgsz, 'iou': 0.7, 'max_det': 300,
              'agnostic_nms': True, 'batch_size': 1, 'device': 'cuda:0', 'half': False, 'seed': 2025,
              'crop_padding_per_side': 0.05, 'crop_format': 'PNG', 'mask_background': False,
              'exif_transpose': True, 'alpha_background_rgb': list(ALPHA_RGB)}
    config_hash = fingerprint(config)
    meta.update(input_sha256=hashes, detector_config=config, detector_config_sha256=config_hash)
    save(folder / 'manifest.json', meta)
    os.environ.update(YOLO_CONFIG_DIR=str(folder / 'settings'), MPLCONFIGDIR=str(folder / 'matplotlib'),
                      YOLO_AUTOINSTALL='false')
    import torch
    from PIL import ImageDraw
    from ultralytics import YOLOE, settings
    from ultralytics.nn.text_model import MobileCLIPTS

    if not torch.cuda.is_available():
        raise RuntimeError('Cần CUDA trong .venv; không tự chuyển CPU.')
    torch.manual_seed(2025)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    meta['resources'] = {'gpu': torch.cuda.get_device_name(0), 'free_vram_before_load': torch.cuda.mem_get_info()[0]}
    integrations = ('sync', 'hub', 'clearml', 'comet', 'dvc', 'mlflow', 'neptune', 'raytune', 'tensorboard', 'wandb')
    settings.update({**{k: False for k in integrations if k in settings},
                     'weights_dir': str(MODELS), 'runs_dir': str(folder)})
    (folder / 'crops').mkdir()
    (folder / 'annotated').mkdir()
    crops, parents, output_paths = [], [], []
    try:
        progress.set('load local YOLOE and encode fixed prompts')
        model = YOLOE(str(MODELS / 'yoloe-v8l-seg.pt'))
        model.model.clip_model = MobileCLIPTS(device=torch.device('cpu'), weight=str(MODELS / 'mobileclip_blt.ts'))
        embeddings = model.model.get_text_pe(PROMPTS, cache_clip_model=True)
        model.set_classes(PROMPTS, embeddings=embeddings)
        del model.model.clip_model
        progress.set('detect and crop traffic signs', len(images))
        for parent in images:
            image = rgb_image(ROOT / parent['path'])
            result = model.predict(source=image, device=0, conf=args.conf, imgsz=args.imgsz, iou=0.7,
                                   agnostic_nms=True, max_det=300, half=False, save=False, verbose=False)[0]
            boxes = []
            for box in result.boxes:
                coords = [round(float(v), 6) for v in box.xyxy[0].cpu().tolist()]
                confidence, class_id = float(box.conf[0].cpu()), int(box.cls[0].cpu())
                x1, y1, x2, y2 = coords
                if (not all(math.isfinite(v) for v in coords + [confidence]) or not 0 <= class_id < len(PROMPTS)
                        or not 0 < confidence <= 1 or not 0 <= x1 < x2 <= image.width
                        or not 0 <= y1 < y2 <= image.height):
                    raise ValueError(f'YOLOE trả bbox/score không hợp lệ: {parent["image_id"]}')
                boxes.append((coords, confidence, class_id))
            boxes.sort(key=lambda b: (b[0][1], b[0][0], b[0][3], b[0][2], -b[1], b[2]))
            annotated, crop_ids = image.copy(), []
            draw = ImageDraw.Draw(annotated)
            for coords, confidence, class_id in boxes:
                x1, y1, x2, y2 = coords
                px, py = (x2 - x1) * 0.05, (y2 - y1) * 0.05
                bounds = [max(0, math.floor(x1 - px)), max(0, math.floor(y1 - py)),
                          min(image.width, math.ceil(x2 + px)), min(image.height, math.ceil(y2 + py))]
                crop = image.crop(bounds)
                cid = 'crop:' + fingerprint([parent['image_id'], config_hash, coords, class_id])
                path = folder / 'crops' / (cid.split(':')[1] + '.png')
                crop.save(path, format='PNG')
                row = {'crop_id': cid, 'parent_image_id': parent['image_id'], 'parent_path': parent['path'],
                       'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path), 'pixel_sha256': pixel_hash(crop),
                       'xyxy': coords, 'crop_xyxy': bounds, 'width': crop.width, 'height': crop.height,
                       'confidence': confidence, 'prompt_id': class_id, 'prompt': PROMPTS[class_id]}
                if 'citations' in parent:
                    row['citations'] = parent['citations']
                crops.append(row)
                crop_ids.append(cid)
                output_paths.append(path)
                draw.rectangle(coords, outline='red', width=3)
                draw.text((x1, max(0, y1 - 14)), f'{len(crop_ids)}: {confidence:.3f}', fill='red')
            path = folder / 'annotated' / (parent['sha256'] + '.png')
            annotated.save(path, format='PNG')
            output_paths.append(path)
            parents.append({**parent, 'width': image.width, 'height': image.height, 'crop_ids': crop_ids,
                            'status': 'detected' if crop_ids else 'no_detection',
                            'max_det_reached': len(boxes) == config['max_det'], 'annotated_path': str(path.relative_to(ROOT))})
            progress.done += 1
    finally:
        meta['resources'].update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                                 peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    crops.sort(key=lambda r: r['crop_id'])
    if len({r['crop_id'] for r in crops}) != len(crops):
        raise ValueError('Trùng crop ID; kiểm tra bbox sau NMS.')
    write_lines(folder / 'images.jsonl', parents)
    write_lines(folder / 'crops.jsonl', crops)
    names = ['images.jsonl', 'crops.jsonl']
    if queries is not None:
        lookup = {p['image_id']: p['crop_ids'] for p in parents}
        write_lines(folder / 'queries.jsonl', [{**q, 'crop_ids': lookup[q['image_id']]} for q in queries])
        names.append('queries.jsonl')
    counts = Counter(len(p['crop_ids']) for p in parents)
    inventory = {'images': len(parents), 'crops': len(crops), 'crop_count_histogram': dict(sorted(counts.items())),
                 'no_detection_images': [p['image_id'] for p in parents if not p['crop_ids']],
                 'max_det_reached_images': [p['image_id'] for p in parents if p['max_det_reached']]}
    if queries is None:
        before = {(c['law_id'], c['article_id']) for p in parents for c in p['citations']}
        after = {(c['law_id'], c['article_id']) for r in crops for c in r['citations']}
        inventory.update(citations_before=len(before), citations_after=len(after),
                         citations_lost=[{'law_id': l, 'article_id': a} for l, a in sorted(before - after)])
    meta.update(inventory=inventory, output_sha256=hashes_of(output_paths + [folder / n for n in names]))
    validate_crops(parents, crops, folder)
    print(f'[crop] {len(parents)} images; {len(crops)} signs; no_detection={counts.get(0, 0)}', flush=True)


def validate_crops(parents, crops, folder):
    lookup = {p['image_id']: p for p in parents}
    by_id = {r['crop_id']: r for r in crops}
    if (not parents or len(lookup) != len(parents) or len(by_id) != len(crops)
            or crops != sorted(crops, key=lambda r: r['crop_id'])):
        raise ValueError('Mapping ảnh/crop rỗng hoặc trùng/sai thứ tự ID.')
    seen = set()
    for parent in parents:
        ids = parent['crop_ids']
        if (len(set(ids)) != len(ids) or parent['status'] != ('detected' if ids else 'no_detection')
                or not isinstance(parent['width'], int) or not isinstance(parent['height'], int)
                or min(parent['width'], parent['height']) <= 0):
            raise ValueError('Trạng thái/kích thước ảnh cha không hợp lệ.')
        for cid in ids:
            row = by_id[cid]
            b, p = row['crop_xyxy'], row['xyxy']
            if (row['parent_image_id'] != parent['image_id'] or cid in seen
                    or row['parent_path'] != parent['path'] or len(b) != 4 or len(p) != 4
                    or not all(isinstance(v, int) for v in b) or not all(math.isfinite(v) for v in p)
                    or not 0 <= b[0] <= p[0] < p[2] <= b[2] <= parent['width']
                    or not 0 <= b[1] <= p[1] < p[3] <= b[3] <= parent['height']
                    or (row['width'], row['height']) != (b[2] - b[0], b[3] - b[1])
                    or not 0 <= row['prompt_id'] < len(PROMPTS) or row['prompt'] != PROMPTS[row['prompt_id']]
                    or not math.isfinite(row['confidence']) or not 0 < row['confidence'] <= 1
                    or not (ROOT / row['path']).resolve().is_relative_to((folder / 'crops').resolve())
                    or ('citations' in parent and row.get('citations') != parent['citations'])):
                raise ValueError(f'Crop/mapping/bbox không hợp lệ: {cid}')
            seen.add(cid)
    if seen != set(by_id):
        raise ValueError('Có crop không thuộc inventory ảnh cha.')


def load_cropping(name):
    names = ['images.jsonl', 'crops.jsonl'] + (['queries.jsonl'] if name == 'query_cropping_dev' else [])
    meta, folder, hashes = load_step(name, names)
    if meta.get('detector_config_sha256') != fingerprint(meta['detector_config']):
        raise ValueError('Fingerprint cấu hình detector sai.')
    parents, crops = read_lines(folder / 'images.jsonl'), read_lines(folder / 'crops.jsonl')
    validate_crops(parents, crops, folder)
    required = {str((ROOT / r['path']).relative_to(ROOT)) for r in crops}
    required.update(p['annotated_path'] for p in parents)
    if not required <= set(meta['output_sha256']):
        raise ValueError('Manifest thiếu hash crop/annotated.')
    queries = read_lines(folder / 'queries.jsonl') if name == 'query_cropping_dev' else None
    if queries is not None:
        lookup = {p['image_id']: p['crop_ids'] for p in parents}
        if (len({q['query_id'] for q in queries}) != len(queries)
                or any(q['crop_ids'] != lookup[q['image_id']] for q in queries)):
            raise ValueError('Query mapping không khớp crop của ảnh cha.')
    return meta, parents, crops, queries, hashes


def crop_corpus(args, folder, meta, progress):
    images, _, hashes = corpus_inventory(progress)
    detect_images(args, folder, meta, progress, images, None, hashes)


def crop_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--overwrite', action='store_true')
    return parser


def validate_arguments(parser, args):
    if not math.isfinite(args.conf) or not 0 < args.conf < 1 or args.imgsz <= 0 or args.imgsz % 32:
        parser.error('conf phải nằm trong (0,1); imgsz phải dương và chia hết cho 32.')


def main():
    parser = crop_arguments()
    args = parser.parse_args()
    validate_arguments(parser, args)
    run_step(args, 'corpus_cropping', crop_corpus, Path(__file__))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()

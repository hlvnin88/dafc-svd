"""Bước 3: Joern extractor (F2–F7)."""
import os, json, subprocess, tempfile, time
import pandas as pd
from tqdm import tqdm
from config import DATA_DIR, JOERN_PATH, JOERN_TIMEOUT, JOERN_SINKS
from src.utils import ensure_dir


class JoernExtractor:
    def __init__(self, joern_path=JOERN_PATH, timeout=JOERN_TIMEOUT):
        self.joern_path = joern_path
        self.timeout = timeout
        self._check()

    def _check(self):
        try:
            r = subprocess.run([self.joern_path, '--version'],
                               capture_output=True, text=True, timeout=20)
            if r.returncode == 0:
                print(f"✅ Joern: {r.stdout.strip()}")
            else:
                print(f"⚠️ Joern err: {r.stderr}")
        except Exception as e:
            print(f"❌ Joern not found at '{self.joern_path}': {e}")
            raise RuntimeError("Joern required for F2–F7")

    def _parse(self, src, wd):
        cpg = os.path.join(wd, 'cpg.bin')
        r = subprocess.run([self.joern_path, 'parse', src, '--output', cpg],
                           capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0:
            raise RuntimeError(f"Joern parse failed: {r.stderr[:300]}")
        return cpg

    def _export(self, cpg, wd, fmt='csv'):
        out = os.path.join(wd, 'export')
        r = subprocess.run(['joern-export', '--cpg', cpg, '--out', out,
                            '--format', fmt],
                           capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0:
            raise RuntimeError(f"Joern export failed: {r.stderr[:300]}")
        return out

    def _slice(self, cpg, wd):
        out = os.path.join(wd, 'slices'); os.makedirs(out, exist_ok=True)
        r = subprocess.run(['joern-slice', 'data-flow', '--cpg', cpg,
                            '--sinks', ','.join(JOERN_SINKS), '--output', out],
                           capture_output=True, text=True, timeout=self.timeout)
        return out if r.returncode == 0 else None

    def extract_one(self, code, sid, skip_slice=False):
        """skip_slice=True: bỏ bước slice (dùng cho Juliet/Big-Vul)."""
        res = {'nodes': None, 'rels': None, 'slices': [], 'status': 'fail', 'err': ''}
        with tempfile.TemporaryDirectory() as wd:
            ext = '.cpp' if ('class ' in code or 'template' in code) else '.c'
            src = os.path.join(wd, f's_{sid}{ext}')
            try:
                with open(src, 'w', encoding='utf-8') as f: f.write(code)
                cpg = self._parse(src, wd)
                exp = self._export(cpg, wd, 'csv')
                for fn in os.listdir(exp):
                    if fn.endswith('nodes.csv') or fn == 'nodes.csv':
                        res['nodes'] = pd.read_csv(os.path.join(exp, fn))
                    elif fn.endswith('rels.csv') or fn == 'rels.csv':
                        res['rels'] = pd.read_csv(os.path.join(exp, fn))
                if not skip_slice:
                    sd = self._slice(cpg, wd)
                    if sd:
                        for fn in os.listdir(sd):
                            if fn.endswith('.json'):
                                with open(os.path.join(sd, fn)) as f:
                                    try: res['slices'].append(json.load(f))
                                    except: pass
                res['status'] = 'ok'
            except Exception as e:
                res['err'] = str(e)[:200]
        return res

    def extract_dataset(self, df, name, skip_slice=False):
        cache_dir = ensure_dir(DATA_DIR / name / 'joern_raw')
        logs = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Joern {name}"):
            sid = str(row['sample_id'])
            cp = cache_dir / f'{sid}.json'
            if cp.exists():
                logs.append({'sample_id': sid, 'status': 'cached', 'time_sec': 0.0})
                continue
            t0 = time.time()
            r = self.extract_one(str(row['source_code']), sid, skip_slice=skip_slice)
            payload = {'status': r['status'], 'err': r['err'],
                       'time_sec': time.time() - t0,
                       'nodes': r['nodes'].to_dict('list') if r['nodes'] is not None else None,
                       'rels':  r['rels'].to_dict('list')  if r['rels']  is not None else None,
                       'slices': r['slices']}
            with open(cp, 'w') as f: json.dump(payload, f)
            logs.append({'sample_id': sid, 'status': r['status'],
                         'err': r['err'], 'time_sec': payload['time_sec']})
        return logs
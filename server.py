
# server.py
# Auto-detect FastAPI server that reuses an existing TF/Keras prediction function in the project.
import os
import io
import sys
import uuid
import json
import inspect
import importlib.util
from pathlib import Path
from typing import Callable, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Local helpers (we implement a small tts/pdf generator below as separate modules)
import tts as tts_module
import pdf_generator as pdf_module

# ------------------- configuration -------------------
PROJECT_ROOT = Path(__file__).parent
STATIC_DIR = PROJECT_ROOT / "static"
REPORTS_DIR = STATIC_DIR / "reports"
AUDIO_DIR = STATIC_DIR / "audio"
UPLOADS_DIR = STATIC_DIR / "uploads"

for d in (STATIC_DIR, REPORTS_DIR, AUDIO_DIR, UPLOADS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# CORS origins (set via env var if needed)
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

app = FastAPI(title="Auto-detect Brain Tumor Prediction Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ------------------- auto-detection utilities -------------------
PRED_FN_NAMES = [
    "predict_and_explain",
    "explain_image",
    "predict_image",
    "predict",
    "run_prediction",
]

MODEL_LOADER_NAMES = [
    "load_model",
    "_load_model",
    "build_model",
]


class Detector:
    """Detects and wraps prediction code inside the project."""

    def __init__(self, root: Path):
        self.root = root
        self.pred_module = None
        self.pred_fn: Optional[Callable] = None
        self.model_loader: Optional[Callable] = None
        self.model = None
        self.info = {}

    def scan(self):
        """Scan .py files in project root (recursively) and import modules to find prediction functions."""
        # Skip server files
        skip_names = {"server.py", "tts.py", "pdf_generator.py"}
        candidates = []
        for p in self.root.rglob("*.py"):
            if p.name in skip_names:
                continue
            candidates.append(p)

        # Try importing each module and inspect
        for path in candidates:
            try:
                mod = self._import_path(path)
            except Exception:
                continue
            # look for pred fn
            for name in PRED_FN_NAMES:
                fn = getattr(mod, name, None)
                if callable(fn):
                    self.pred_module = mod
                    self.pred_fn = fn
                    self.info['pred_module'] = path.name
                    self.info['pred_fn'] = name
                    break
            if self.pred_fn:
                # try find model loader
                for mname in MODEL_LOADER_NAMES:
                    mfn = getattr(mod, mname, None)
                    if callable(mfn):
                        self.model_loader = mfn
                        self.info['model_loader_module'] = path.name
                        self.info['model_loader_name'] = mname
                        break
                # or search other functions in other modules for model loader
                if not self.model_loader:
                    self._find_model_loader_elsewhere(candidates)
                return True
        return False

    def _find_model_loader_elsewhere(self, candidates):
        for path in candidates:
            try:
                mod = self._import_path(path)
            except Exception:
                continue
            for mname in MODEL_LOADER_NAMES:
                mfn = getattr(mod, mname, None)
                if callable(mfn):
                    self.model_loader = mfn
                    self.info['model_loader_module'] = path.name
                    self.info['model_loader_name'] = mname
                    return

    def _import_path(self, path: Path):
        """Dynamically import a python file as a module and return it."""
        name = "autodetect_" + str(uuid.uuid4()).replace('-', '')
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    def try_load_model(self):
        """If a model loader exists, call it and cache the model. If loader requires args, we try without args.
        If no loader, we do nothing and rely on the prediction function to self-load.
        """
        if not self.model_loader:
            return False
        try:
            # try calling with no args
            m = self.model_loader()
            self.model = m
            self.info['model_loaded'] = True
            return True
        except TypeError:
            # loader needs args -- give up and let prediction function handle it
            self.info['model_loaded'] = False
            return False
        except Exception as e:
            self.info['model_loaded'] = False
            self.info['model_load_error'] = str(e)
            return False

    def call_predict(self, image_path: str) -> Dict[str, Any]:
        """Call the discovered prediction function with best-effort argument mapping.
        Accepts an image file path and returns a dictionary with keys we expect (flexible).
        """
        if not self.pred_fn:
            raise RuntimeError("No prediction function detected")


        fn = self.pred_fn
        sig = inspect.signature(fn)
        kwargs = {}
        args = []

        # heuristics: if function accepts 'image' or 'image_path' or 'img' -> pass path
        for pname, p in sig.parameters.items():
            n = pname.lower()
            if 'image' in n or 'img' in n or 'path' in n:
                args.append(image_path)
            elif 'model' in n and self.model is not None:
                args.append(self.model)
            elif p.default is not inspect._empty:
                # skip optional
                pass
            else:
                # can't satisfy non-optional param; we will try calling with only image arg
                pass

        # Call and capture return value
        try:
            res = fn(*args) if args else fn(image_path)
        except TypeError:
            # try passing keyword if function expects named args e.g., predict(image_path=...)
            try:
                res = fn(image_path=image_path)
            except Exception as e:
                raise
        # normalize output into a dict
        return self._normalize_result(res)

    def _normalize_result(self, res: Any) -> Dict[str, Any]:
        """Take arbitrary return value and turn into a dict with common keys.
        Expected forms:
         - dict with keys 'pred_label','pred_prob','overlay_path', ...
         - tuple/list like (label, prob, overlay_path)
         - object with attributes
        """
        out = {}
        if isinstance(res, dict):
            out.update(res)
        elif isinstance(res, (list, tuple)):
            # common order: label, prob, overlay
            if len(res) >= 1:
                out['pred_label'] = res[0]
            if len(res) >= 2:
                out['pred_prob'] = res[1]
            if len(res) >= 3:
                out['overlay_path'] = res[2]
        else:
            # try attribute access
            for attr in ('pred_label', 'label', 'pred', 'prediction'):
                if hasattr(res, attr):
                    out['pred_label'] = getattr(res, attr)
            for attr in ('pred_prob', 'prob', 'confidence'):
                if hasattr(res, attr):
                    out['pred_prob'] = float(getattr(res, attr))
        return out


# Initialize detector on startup
DETECTOR = Detector(PROJECT_ROOT)
found = DETECTOR.scan()
if not found:
    print("[WARNING] No prediction function found in project files. Place your prediction script in the project.")
else:
    print(f"[INFO] Detected prediction function: {DETECTOR.info}")
    DETECTOR.try_load_model()

# ------------------- helpers: mapping and persistence -------------------

MALIGNANCY_MAP = {
    'glioma': 'Malignant',
    'meningioma': 'Benign',
    'pituitary': 'Benign',
    'no tumor': 'Benign',
    'notumor': 'Benign',
    'no_tumor': 'Benign',
    'not tumor': 'Benign'
}

def normalize_label(label: str) -> str:
    if label is None:
        return 'Unknown'
    lab = label.strip().lower().replace(' ', '').replace('-', '')
    # map fuzzy terms
    if 'glioma' in lab:
        return 'glioma'
    if 'meningioma' in lab:
        return 'meningioma'
    if 'pituitary' in lab:
        return 'pituitary'
    if 'no' in lab or 'not' in lab or 'normal' in lab or 'none' in lab:
        return 'no tumor'
    # fallback: original lower
    return label.strip().lower()

def malignancy_from_label(norm_label: str) -> str:
    return MALIGNANCY_MAP.get(norm_label, 'Benign')

def save_report_metadata(report_id: str, metadata: Dict[str, Any]):
    path = REPORTS_DIR / f"{report_id}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    return str(path)

def load_report_metadata(report_id: str) -> Dict[str, Any]:
    path = REPORTS_DIR / f"{report_id}.json"
    if not path.exists():
        raise FileNotFoundError
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ------------------- API endpoints -------------------

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """Accept an MRI image, call discovered prediction function, generate TTS + PDF, return structured JSON.

    This endpoint attempts to be robust to different prediction function signatures by saving the
    uploaded file and passing its path to the detected function.
    """
    # ensure detector ready
    if DETECTOR.pred_fn is None:
        raise HTTPException(status_code=500, detail="No prediction function detected on server startup")

    # save upload to disk temporarily
    suffix = Path(file.filename).suffix or '.jpg'
    upload_id = str(uuid.uuid4())
    upload_path = UPLOADS_DIR / f"{upload_id}{suffix}"
    with open(upload_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    # call prediction
    try:
        raw = DETECTOR.call_predict(str(upload_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # extract fields
    label = raw.get('pred_label') or raw.get('label') or raw.get('prediction')
    prob = raw.get('pred_prob') or raw.get('prob') or raw.get('confidence') or 0.0
    overlay_path = raw.get('overlay_path') or raw.get('gradcam') or raw.get('gradcam_path')

    norm_label = normalize_label(str(label) if label else '')
    malignancy = malignancy_from_label(norm_label)

    # textual summary
    summary = (
        f"Automated MRI analysis indicates {norm_label.title()} with confidence {float(prob):.2%}. "
        f"Clinical mapping suggests this is '{malignancy}'. This AI output is informational and should be reviewed by a radiologist."
    )

    # ensure overlay image exists; if not, try to create one from returned overlay_path
    gradcam_b64 = None
    gradcam_url = None
    target_overlay = None
    if overlay_path and Path(overlay_path).exists():
        # copy it to reports directory (use copy to preserve original)
        report_id = str(uuid.uuid4())
        target_overlay = REPORTS_DIR / f"{report_id}_gradcam.png"
        # copy file content
        with open(overlay_path, 'rb') as src, open(target_overlay, 'wb') as dst:
            dst.write(src.read())
        gradcam_url = f"/static/reports/{target_overlay.name}"
        with open(target_overlay, 'rb') as f:
            gradcam_b64 = (f.read()).hex()  # hex as light-weight fallback; caller can decode
    else:
        report_id = str(uuid.uuid4())

    # TTS generation
    audio_path = AUDIO_DIR / f"{report_id}.mp3"
    try:
        tts_module.synthesize_tts(summary, str(audio_path))
        audio_url = f"/static/audio/{audio_path.name}"
    except Exception as e:
        audio_url = None

    # PDF generation
    pdf_path = REPORTS_DIR / f"{report_id}.pdf"
    try:
        pdf_module.build_pdf(str(pdf_path), patient_id=None, tumor_type=norm_label, benign_malignant=malignancy, confidence=float(prob), summary=summary, gradcam_path=str(target_overlay) if target_overlay else None, created_at=None)
        pdf_url = f"/static/reports/{pdf_path.name}"
    except Exception:
        pdf_url = None

    metadata = {
        'report_id': report_id,
        'uploaded_filename': file.filename,
        'pred_label': norm_label,
        'malignancy': malignancy,
        'confidence': float(prob),
        'summary': summary,
        'gradcam_url': gradcam_url,
        'audio_url': audio_url,
        'pdf_url': pdf_url,
    }
    save_report_metadata(report_id, metadata)

    return metadata

@app.get('/download/pdf/{report_id}')
def download_pdf(report_id: str):
    path = REPORTS_DIR / f"{report_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail='PDF not found')
    return FileResponse(str(path), media_type='application/pdf', filename=path.name)

@app.get('/download/audio/{report_id}')
def download_audio(report_id: str):
    path = AUDIO_DIR / f"{report_id}.mp3"
    if not path.exists():
        raise HTTPException(status_code=404, detail='Audio not found')
    return FileResponse(str(path), media_type='audio/mpeg', filename=path.name)

@app.get('/reports')
def list_reports():
    items = []
    for p in REPORTS_DIR.glob('*.json'):
        with open(p, 'r', encoding='utf-8') as f:
            items.append(json.load(f))
    # newest first
    items.sort(key=lambda x: x.get('report_id'), reverse=True)
    return {'reports': items}

# ------------------- static mount for client convenience -------------------
from fastapi.staticfiles import StaticFiles
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')

# ------------------- server startup message -------------------
@app.on_event('startup')
def on_startup():
    print('[STARTUP] Auto-detect server ready. Prediction function:', DETECTOR.info)

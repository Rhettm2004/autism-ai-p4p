"""Run the supplied EAIP-DARV artifacts behind their documented HTTP contract.

The model source and weights stay outside Git. Set EAIP_PROJECT_DIR to the
extracted deployment folder before starting this service.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import zipfile

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: dict[str, Any]


predictor = None
load_error: str | None = None


def _portable_contract(source: Path, project: Path, output: Path) -> Path:
    contract = json.loads(source.read_text())
    if contract.get('module') == 'module2_ASSL_NB_CNN_DDT_CDL':
        model_dir = project / 'model_artifacts/module2_ASSL_outputs/saved_model'
        contract['paths']['preprocessor'] = str(model_dir / 'module2_preprocessor.joblib')
        contract['paths']['cnn_weights'] = str(model_dir / 'module2_cnn_final.pt')
    else:
        model_dir = project / 'model_artifacts/module3_outputs/saved_model'
        contract['paths']['preprocessor'] = str(model_dir / 'module3_preprocessor.joblib')
        contract['paths']['final_cnn'] = str(model_dir / 'module3_final_cnn.keras')
    output.write_text(json.dumps(contract))
    return output


def _load_module1_compat(core, model_path: Path, prep_path: Path):
    """Load the supplied Keras 2.13 HDF5 artifact on current macOS runtimes.

    The file contains a valid model config and weights, but Keras' legacy HDF5
    loader fails to associate its nested weight groups with the layers.  Keep
    the supplied architecture and tensors intact and only perform that mapping
    explicitly when the original loader fails.
    """
    try:
        return core.load_module1(model_path, prep_path)
    except ValueError as original_error:
        import h5py
        from tensorflow import keras

        prep = json.loads(prep_path.read_text())
        inputs = keras.Input(shape=(len(prep['columns']), 1), name='input_1')
        value = keras.layers.Conv1D(
            32, 3, padding='same', activation='relu', name='conv1d')(inputs)
        value = keras.layers.MaxPooling1D(2, name='max_pooling1d')(value)
        value = keras.layers.Conv1D(
            64, 3, padding='same', activation='relu', name='conv1d_1')(value)
        value = keras.layers.MaxPooling1D(2, name='max_pooling1d_1')(value)
        value = keras.layers.Flatten(name='flatten')(value)
        value = keras.layers.Dense(64, activation='relu', name='dense')(value)
        value = keras.layers.Dropout(0.30, name='dropout')(value)
        outputs = keras.layers.Dense(1, activation='sigmoid', name='dense_1')(value)
        model = keras.Model(inputs, outputs)
        try:
            with h5py.File(model_path, 'r') as handle:
                weights = handle['model_weights']
                for name in ('conv1d', 'conv1d_1', 'dense', 'dense_1'):
                    group = weights[name][name]
                    model.get_layer(name).set_weights([
                        group['kernel:0'][()], group['bias:0'][()],
                    ])
        except Exception:
            raise original_error
        return {'model': model, 'prep': prep}


def _load_module3_compat(core, contract_path: Path):
    """Load Module 3 when its Windows-authored archive uses backslashes.

    Keras 2.13 wrote checkpoint group names containing Windows separators into
    this particular `.keras` archive.  macOS Keras cannot match those groups,
    so reconstruct the unchanged published network and attach its tensors.
    """
    try:
        return core.load_module3(contract_path)
    except ValueError as original_error:
        import h5py
        import joblib
        from tensorflow import keras

        contract = json.loads(contract_path.read_text())
        pre = joblib.load(contract['paths']['preprocessor'])
        raw_cols = (
            list(pre.feature_names_in_)
            if hasattr(pre, 'feature_names_in_') else None
        )
        transformed_width = len(pre.get_feature_names_out())
        model = keras.Sequential([
            keras.layers.Input(shape=(transformed_width, 1), name='input_1'),
            keras.layers.Conv1D(
                32, 3, activation='relu', padding='same', name='conv1d'),
            keras.layers.MaxPooling1D(2, name='max_pooling1d'),
            keras.layers.Conv1D(
                64, 3, activation='relu', padding='same', name='conv1d_1'),
            keras.layers.GlobalAveragePooling1D(
                name='global_average_pooling1d'),
            keras.layers.Dense(64, activation='relu', name='dense'),
            keras.layers.Dropout(0.25, name='dropout'),
            keras.layers.Dense(1, activation='sigmoid', name='dense_1'),
        ])
        try:
            with zipfile.ZipFile(contract['paths']['final_cnn']) as archive:
                temporary = Path(tempfile.mkdtemp(prefix='eaip-module3-'))
                weights_path = Path(archive.extract(
                    'model.weights.h5', temporary))
            with h5py.File(weights_path, 'r') as handle:
                mappings = (
                    ('conv1d', 'conv1d'),
                    ('conv1d_1', 'conv1d_2'),
                    ('dense', 'dense'),
                    ('dense_1', 'dense_2'),
                )
                for layer_name, checkpoint_name in mappings:
                    group = handle[
                        f'_layer_checkpoint_dependencies\\{checkpoint_name}/vars']
                    model.get_layer(layer_name).set_weights([
                        group['0'][()], group['1'][()],
                    ])
        except Exception:
            raise original_error
        return {
            'pre': pre,
            'model': model,
            'thr': float(contract.get('prediction', {}).get('threshold', 0.5)),
            'raw_cols': raw_cols,
        }


class IntegratedPredictor:
    """Loads the supplied modules while resolving exported Windows paths."""

    def __init__(self, project: Path):
        sys.path.insert(0, str(project))
        import eaip_darv_core as core
        import joblib

        self.core = core
        artifacts = project / 'model_artifacts'
        temporary = Path(tempfile.mkdtemp(prefix='eaip-contracts-'))
        module2_contract = _portable_contract(
            artifacts / 'module2_ASSL_outputs/saved_model/module2_contract.json',
            project,
            temporary / 'module2_contract.json',
        )
        module3_contract = _portable_contract(
            artifacts / 'module3_outputs/saved_model/module3_contract.json',
            project,
            temporary / 'module3_contract.json',
        )
        self.mod1 = _load_module1_compat(
            core,
            artifacts / 'saved_models/module1_screening_cnn.h5',
            artifacts / 'saved_models/module1_screening_preprocess.json',
        )
        self.mod2 = core.load_module2(module2_contract)
        self.mod3 = _load_module3_compat(core, module3_contract)
        if any(module is None for module in (self.mod1, self.mod2, self.mod3)):
            raise RuntimeError('All three supplied EAIP-DARV modules must load.')
        bundle = artifacts / 'eaip_darv_bundle'
        self.calibrators = joblib.load(bundle / 'calibrators.joblib')
        self.config = json.loads((bundle / 'config.json').read_text())
        self._raw_cols = sorted(set(
            self.mod1['prep']['columns']
            + self.mod2['feature_cols']
            + list(self.mod3.get('raw_cols') or [])
        ))

    def expected_raw_columns(self):
        return self._raw_cols

    def predict(self, raw_features: dict[str, Any]):
        import numpy as np
        import pandas as pd

        categorical = dict(raw_features)
        numeric = dict(raw_features)
        numeric['Sex'] = {'f': 0, 'm': 1}.get(str(raw_features.get('Sex')).lower())
        numeric['Jauntice'] = 1 if raw_features.get('Jauntice') == 'yes' else 0
        numeric['FamilyASDHistory'] = (
            1 if raw_features.get('FamilyASDHistory') == 'yes' else 0
        )
        probs_raw = {}
        probability, _ = self.core.predict_module1(
            self.mod1, pd.DataFrame([numeric]))
        probs_raw['M1_screening'] = probability
        probability, _ = self.core.predict_module2(
            self.mod2, pd.DataFrame([categorical]))
        probs_raw['M2_ASSL'] = probability
        probability, _ = self.core.predict_module3(
            self.mod3, pd.DataFrame([categorical]))
        probs_raw['M3_cluster'] = probability

        probs_cal = self.core.apply_calibrators(probs_raw, self.calibrators)
        agreement = self.core.compute_agreement_scores(probs_cal)
        p_eaip = self.core.eaipp_gated_vote(probs_cal, agreement)
        p_darv, disagreement, confidence = self.core.darv_disagreement_aware_vote(
            probs_cal, agreement, lam=self.config['lambda'])
        thresholds = {
            'eaip': self.config['eaip_threshold'],
            'darv_fixed': self.config['darv_fixed_threshold'],
            'darv_tuned': self.config['darv_tuned_threshold'],
        }
        return {
            'per_module_raw_probability': {
                key: float(value[0]) for key, value in probs_raw.items()},
            'per_module_calibrated_probability': {
                key: float(value[0]) for key, value in probs_cal.items()},
            'agreement_scores': {
                key: float(value[0]) for key, value in agreement.items()},
            'eaip_probability': float(p_eaip[0]),
            'darv_probability': float(p_darv[0]),
            'disagreement': float(disagreement[0]),
            'confidence_pi': float(confidence[0]),
            'classification_eaip': int(p_eaip[0] >= thresholds['eaip']),
            'classification_darv_fixed': int(
                p_darv[0] >= thresholds['darv_fixed']),
            'classification_darv_tuned': int(
                p_darv[0] >= thresholds['darv_tuned']),
            'thresholds_used': thresholds,
        }


@asynccontextmanager
async def lifespan(_app):
    global predictor, load_error
    project_dir = os.getenv('EAIP_PROJECT_DIR')
    try:
        if not project_dir:
            raise RuntimeError('EAIP_PROJECT_DIR is not set.')
        predictor = IntegratedPredictor(Path(project_dir).resolve())
    except Exception as exc:
        load_error = f'{type(exc).__name__}: {exc}'
    yield


app = FastAPI(
    title='EAIP-DARV Autism Screening Classifier',
    version='1.0.0-integrated',
    lifespan=lifespan,
)


@app.get('/health')
def health():
    if predictor is None:
        raise HTTPException(503, load_error or 'Model not loaded.')
    return {'status': 'ok', 'model_loaded': True}


@app.get('/schema')
def schema():
    if predictor is None:
        raise HTTPException(503, load_error or 'Model not loaded.')
    return {'expected_raw_columns': predictor.expected_raw_columns()}


@app.post('/predict')
def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(503, load_error or 'Model not loaded.')
    try:
        return predictor.predict(request.features)
    except Exception as exc:
        raise HTTPException(400, f'Prediction failed: {type(exc).__name__}') from exc

import { useState, useEffect, useCallback, useMemo } from 'react';
import { apiClient, ApiError } from '../api/client';
import {
  GeometryValidationResponse,
  ForwardPredictionResponse,
} from '../api/types';
import { useDebounce } from './useDebounce';
import { CanonicalGeometry, CANONICAL_GEOMETRIES } from '../components/forward/GeometrySelector';

export function useForwardAnalysis() {
  // 1. Physical Parameters State
  const [plateLength, setPlateLength] = useState<number>(1000.0);
  const [plateHeight, setPlateHeight] = useState<number>(1000.0);
  const [centralHoleDiameter, setCentralHoleDiameter] = useState<number>(83.333);
  const [defectDiameter, setDefectDiameter] = useState<number>(20.0);
  const [defectX, setDefectX] = useState<number>(300.0);
  const [defectY, setDefectY] = useState<number>(700.0);
  const [appliedTraction, setAppliedTraction] = useState<number>(100.0);
  const [framework, setFramework] = useState<'heterogeneous' | 'dnn' | 'pca'>('heterogeneous');

  // 2. Validation & Prediction State
  const [validation, setValidation] = useState<GeometryValidationResponse | null>(null);
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const [prediction, setPrediction] = useState<ForwardPredictionResponse | null>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  // Snapshot of parameters used for the current prediction to detect staleness
  const [predictedConfigHash, setPredictedConfigHash] = useState<string | null>(null);

  // Compute a hash of current physical parameters
  const currentConfigHash = useMemo(() => {
    return `${plateLength}-${plateHeight}-${centralHoleDiameter.toFixed(2)}-${defectDiameter}-${defectX}-${defectY}-${appliedTraction}-${framework}`;
  }, [plateLength, plateHeight, centralHoleDiameter, defectDiameter, defectX, defectY, appliedTraction, framework]);

  const isStale = Boolean(prediction && predictedConfigHash && currentConfigHash !== predictedConfigHash);

  // Debounced values for geometry validation (180ms delay)
  const debouncedGeom = useDebounce(
    { plateLength, plateHeight, centralHoleDiameter, defectDiameter, defectX, defectY },
    180
  );

  // Run geometry validation against FastAPI endpoint
  const runValidation = useCallback(async () => {
    setIsValidating(true);
    setValidationError(null);
    try {
      const res = await apiClient.validateGeometry({
        plate_length_mm: debouncedGeom.plateLength,
        plate_height_mm: debouncedGeom.plateHeight,
        central_hole_diameter_mm: debouncedGeom.centralHoleDiameter,
        defect_diameter_mm: debouncedGeom.defectDiameter,
        defect_center_x_mm: debouncedGeom.defectX,
        defect_center_y_mm: debouncedGeom.defectY,
      });
      setValidation(res);
    } catch (err: any) {
      setValidationError(err.message || 'Geometry validation request failed.');
    } finally {
      setIsValidating(false);
    }
  }, [debouncedGeom]);

  useEffect(() => {
    runValidation();
  }, [runValidation]);

  // Execute forward prediction
  const runPrediction = useCallback(async () => {
    if (!validation?.can_proceed_to_inference) return;
    setIsPredicting(true);
    setPredictionError(null);

    try {
      const res = await apiClient.predictForward({
        plate_length_mm: plateLength,
        plate_height_mm: plateHeight,
        central_hole_diameter_mm: centralHoleDiameter,
        defect_diameter_mm: defectDiameter,
        defect_center_x_mm: defectX,
        defect_center_y_mm: defectY,
        applied_traction_psi: appliedTraction,
        framework,
      });

      setPrediction(res);
      setPredictedConfigHash(currentConfigHash);
    } catch (err: any) {
      if (err instanceof ApiError) {
        setPredictionError(err.message);
      } else {
        setPredictionError(err.message || 'Prediction execution failed.');
      }
    } finally {
      setIsPredicting(false);
    }
  }, [
    validation,
    plateLength,
    plateHeight,
    centralHoleDiameter,
    defectDiameter,
    defectX,
    defectY,
    appliedTraction,
    framework,
    currentConfigHash,
  ]);

  // Geometry selector handler
  const selectGeometry = (geom: CanonicalGeometry) => {
    setPlateLength(geom.length_mm);
    setPlateHeight(geom.height_mm);
    setCentralHoleDiameter(geom.canonical_dc_mm);
    // Adjust defect Y if it exceeds the new plate height
    if (defectY > geom.height_mm - 20) {
      setDefectY(Math.round(geom.height_mm * 0.7));
    }
  };

  // Reset defaults
  const resetDefaults = () => {
    const defaultGeom = CANONICAL_GEOMETRIES[0]; // 1000x1000
    selectGeometry(defaultGeom);
    setDefectDiameter(20.0);
    setDefectX(300.0);
    setDefectY(700.0);
    setAppliedTraction(100.0);
    setFramework('heterogeneous');
  };

  return {
    // Parameters
    plateLength,
    plateHeight,
    centralHoleDiameter,
    defectDiameter,
    defectX,
    defectY,
    appliedTraction,
    framework,

    // Setters
    setDefectX,
    setDefectY,
    setDefectDiameter,
    setAppliedTraction,
    setFramework,
    selectGeometry,
    resetDefaults,

    // Validation
    validation,
    isValidating,
    validationError,
    canPredict: Boolean(validation?.can_proceed_to_inference),
    hasCollision: Boolean(validation && !validation.is_valid),

    // Prediction
    prediction,
    isPredicting,
    predictionError,
    runPrediction,

    // Stale state
    isStale,
  };
}


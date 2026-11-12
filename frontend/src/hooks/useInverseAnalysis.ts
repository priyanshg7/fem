import { useState, useCallback, useRef } from 'react';
import { apiClient, ApiError } from '../api/client';
import {
  InversePredictionRequest,
  InversePredictionResponse,
  StressSourceProvenance,
  ReferenceGroundTruth,
} from '../api/types';
import { MatrixDiagnostics } from '../utils/matrixValidator';

export function useInverseAnalysis() {
  const [plateLengthMm, setPlateLengthMm] = useState<number>(1000.0);
  const [plateHeightMm, setPlateHeightMm] = useState<number>(700.0);
  const [appliedTractionPsi, setAppliedTractionPsi] = useState<number>(100.0);
  const [stressGrid, setStressGrid] = useState<number[][] | null>(null);
  const [stressProvenance, setStressProvenance] = useState<StressSourceProvenance>('DEMONSTRATION_SAMPLE');
  const [stressDescription, setStressDescription] = useState<string>('Demo 1000x700 mm Confirmatory Field');
  const [diagnostics, setDiagnostics] = useState<MatrixDiagnostics | null>(null);
  const [referenceGroundTruth, setReferenceGroundTruth] = useState<ReferenceGroundTruth | undefined>();

  const [prediction, setPrediction] = useState<InversePredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [isStale, setIsStale] = useState<boolean>(false);

  // Stale parameter tracking
  const lastPredictedInputRef = useRef<string | null>(null);

  const checkStale = useCallback(
    (L: number, H: number, traction: number, grid: number[][] | null) => {
      if (!lastPredictedInputRef.current) return;
      const currentSignature = `${L}_${H}_${traction}_${grid?.length || 0}`;
      setIsStale(currentSignature !== lastPredictedInputRef.current);
    },
    []
  );

  const handleGeometryChange = useCallback(
    (length: number, height: number) => {
      setPlateLengthMm(length);
      setPlateHeightMm(height);
      checkStale(length, height, appliedTractionPsi, stressGrid);
    },
    [appliedTractionPsi, stressGrid, checkStale]
  );

  const handleTractionChange = useCallback(
    (traction: number) => {
      setAppliedTractionPsi(traction);
      checkStale(plateLengthMm, plateHeightMm, traction, stressGrid);
    },
    [plateLengthMm, plateHeightMm, stressGrid, checkStale]
  );

  const handleStressGridChange = useCallback(
    (grid: number[][], diag: MatrixDiagnostics) => {
      setStressGrid(grid);
      setDiagnostics(diag);
      checkStale(plateLengthMm, plateHeightMm, appliedTractionPsi, grid);
    },
    [plateLengthMm, plateHeightMm, appliedTractionPsi, checkStale]
  );

  const handleProvenanceChange = useCallback((prov: StressSourceProvenance, desc: string) => {
    setStressProvenance(prov);
    setStressDescription(desc);
  }, []);

  const executeInverse = useCallback(async () => {
    if (!stressGrid || !diagnostics?.isValid) {
      setError(
        new ApiError(
          400,
          'Cannot execute inverse localization: stress grid is missing or invalid.',
          'INVALID_INPUT'
        )
      );
      return;
    }

    setIsLoading(true);
    setError(null);

    const payload: InversePredictionRequest = {
      plate_length_mm: plateLengthMm,
      plate_height_mm: plateHeightMm,
      applied_traction_psi: appliedTractionPsi,
      stress_grid_64x64: stressGrid,
      return_heatmap: true,
      confidence_abstention_threshold_mm: 9.07,
    };

    try {
      const response = await apiClient.predictInverse(payload);
      setPrediction(response);
      setIsStale(false);
      lastPredictedInputRef.current = `${plateLengthMm}_${plateHeightMm}_${appliedTractionPsi}_${stressGrid.length}`;
    } catch (err: any) {
      if (err instanceof ApiError) {
        setError(err);
      } else {
        setError(new ApiError(500, err.message || 'Inference execution failed.', 'INFERENCE_ERROR'));
      }
    } finally {
      setIsLoading(false);
    }
  }, [plateLengthMm, plateHeightMm, appliedTractionPsi, stressGrid, diagnostics]);

  return {
    plateLengthMm,
    plateHeightMm,
    appliedTractionPsi,
    stressGrid,
    stressProvenance,
    stressDescription,
    diagnostics,
    referenceGroundTruth,
    prediction,
    isLoading,
    error,
    isStale,
    handleGeometryChange,
    handleTractionChange,
    handleStressGridChange,
    handleProvenanceChange,
    setReferenceGroundTruth,
    executeInverse,
  };
}

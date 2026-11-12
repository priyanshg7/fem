import React, { useState, useEffect } from 'react';
import { Play, FileText } from 'lucide-react';
import { Header } from './components/layout/Header';
import { Navigation, WorkspaceType } from './components/layout/Navigation';
import { Footer } from './components/layout/Footer';
import { Card } from './components/common/Card';
import { Button } from './components/common/Button';
import { Alert } from './components/common/Alert';
import { GeometrySelector } from './components/forward/GeometrySelector';
import { PlateViewer } from './components/forward/PlateViewer';
import { ParameterControls } from './components/forward/ParameterControls';
import { ClearancePanel } from './components/forward/ClearancePanel';
import { ResultsPanel } from './components/forward/ResultsPanel';
import { StaleNotice } from './components/forward/StaleNotice';
import { InverseWorkspace } from './components/inverse/InverseWorkspace';
import { CaseHistoryWorkspace } from './components/history/CaseHistoryWorkspace';
import { ReportViewer } from './components/reporting/ReportViewer';
import { useBackendStatus } from './hooks/useBackendStatus';
import { useForwardAnalysis } from './hooks/useForwardAnalysis';
import { caseHistoryService, generateCaseId } from './services/caseHistory';
import { CaseRecord } from './api/types';

export const App: React.FC = () => {
  const [currentWorkspace, setCurrentWorkspace] = useState<WorkspaceType>('forward');
  const [activeReportCase, setActiveReportCase] = useState<CaseRecord | null>(null);
  const [caseCount, setCaseCount] = useState<number>(0);

  // Update case count on mount and workspace change
  useEffect(() => {
    setCaseCount(caseHistoryService.getAllCases().length);
  }, [currentWorkspace, activeReportCase]);

  // Backend connection hook
  const { isConnected, isReady, health, error: backendError } = useBackendStatus();

  // Forward analysis hook
  const {
    plateLength,
    plateHeight,
    centralHoleDiameter,
    defectDiameter,
    defectX,
    defectY,
    appliedTraction,
    framework,
    setDefectX,
    setDefectY,
    setDefectDiameter,
    setAppliedTraction,
    setFramework,
    selectGeometry,
    resetDefaults,
    validation,
    isValidating,
    canPredict,
    hasCollision,
    prediction,
    isPredicting,
    predictionError,
    runPrediction,
    isStale,
  } = useForwardAnalysis();

  // Save Forward Case & open report
  const handleSaveForwardCase = async () => {
    if (!prediction) return;
    const caseId = generateCaseId('FORWARD');
    const newRecord = await caseHistoryService.saveCase({
      case_id: caseId,
      analysis_type: 'FORWARD',
      title: `Forward Stress Concentration (${plateLength}×${plateHeight} mm)`,
      created_at: new Date().toISOString(),
      api_version: '1.0.0',
      frontend_version: '1.0.0',
      provenance: 'ML_SURROGATE',
      forward_request: {
        plate_length_mm: plateLength,
        plate_height_mm: plateHeight,
        central_hole_diameter_mm: centralHoleDiameter,
        defect_diameter_mm: defectDiameter,
        defect_center_x_mm: defectX,
        defect_center_y_mm: defectY,
        applied_traction_psi: appliedTraction,
        framework,
      },
      forward_response: prediction,
      is_stale: isStale,
    });
    setCaseCount(caseHistoryService.getAllCases().length);
    setActiveReportCase(newRecord);
  };

  return (
    <div className="app-container">
      {/* Top Application Header */}
      <Header
        backendConnected={isConnected}
        backendReady={isReady}
        activeModelId={
          currentWorkspace === 'inverse'
            ? health?.active_inverse_model || 'inverse_unet_d5_stage3d'
            : health?.active_forward_model || 'forward_ensemble_stage3b'
        }
        activeModelVersion={currentWorkspace === 'inverse' ? '3.4.0-frozen' : '3.2.0-frozen'}
      />

      {/* Main Navigation Tabs */}
      <Navigation
        currentWorkspace={currentWorkspace}
        onSelectWorkspace={setCurrentWorkspace}
        caseCount={caseCount}
      />

      {/* Backend Disconnection Banner if offline */}
      {!isConnected && (
        <div style={{ padding: '10px 24px 0 24px' }}>
          <Alert type="danger" code="BACKEND_UNAVAILABLE">
            {backendError ? `${backendError}. ` : ''}Cannot reach SimuStruct Backend service at{' '}
            <code>http://localhost:8000/api/v1</code>. Please launch the backend with{' '}
            <code>python backend/run_server.py</code> to execute surrogate predictions.
          </Alert>
        </div>
      )}

      {/* Workspace Content Router */}
      <main style={{ padding: '20px 24px', maxWidth: '1600px', margin: '0 auto', width: '100%' }}>
        {/* Workspace 1: Forward Stress Analysis (Stage 5B) */}
        {currentWorkspace === 'forward' && (
          <div className="workspace-grid" style={{ padding: 0 }}>
            {/* Left Column: Visual Geometry Editor & Clearance Analytics */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <Card title="Plate Geometry Configuration">
                <GeometrySelector
                  selectedHeight={plateHeight}
                  onSelectGeometry={selectGeometry}
                  disabled={isPredicting}
                />
              </Card>

              <Card title="Interactive Specimen Visualization">
                <PlateViewer
                  plateLength={plateLength}
                  plateHeight={plateHeight}
                  centralHoleDiameter={centralHoleDiameter}
                  defectDiameter={defectDiameter}
                  defectX={defectX}
                  defectY={defectY}
                  onChangeDefectPosition={(x, y) => {
                    setDefectX(x);
                    setDefectY(y);
                  }}
                  isValid={Boolean(validation?.is_valid)}
                  hasCollision={hasCollision}
                  ligamentClearance={validation?.clearances?.hole_to_hole_clearance_mm}
                  minBoundaryClearance={validation?.clearances?.min_boundary_clearance_mm}
                  readOnly={isPredicting}
                />
              </Card>

              <Card title="Geometric Non-Penetration Validation">
                <ClearancePanel validation={validation} isValidating={isValidating} />
              </Card>
            </div>

            {/* Right Column: Parameter Controls & Prediction Results */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <Card title="Defect Controls & Loading">
                <ParameterControls
                  plateLength={plateLength}
                  plateHeight={plateHeight}
                  defectX={defectX}
                  defectY={defectY}
                  defectDiameter={defectDiameter}
                  appliedTraction={appliedTraction}
                  framework={framework}
                  onChangeX={setDefectX}
                  onChangeY={setDefectY}
                  onChangeDiameter={setDefectDiameter}
                  onChangeTraction={setAppliedTraction}
                  onChangeFramework={setFramework}
                  onResetDefaults={resetDefaults}
                  disabled={isPredicting}
                />

                <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <Button
                    variant="primary"
                    isLoading={isPredicting}
                    disabled={!canPredict || !isConnected || isPredicting}
                    onClick={runPrediction}
                    icon={<Play size={16} fill="currentColor" />}
                    style={{ width: '100%', padding: '12px 18px', fontSize: '0.9rem' }}
                  >
                    Run Forward Surrogate Analysis
                  </Button>

                  {!canPredict && (
                    <div style={{ fontSize: '0.7rem', color: 'var(--color-danger)', textAlign: 'center' }}>
                      Prediction blocked: resolve physical collisions or domain errors.
                    </div>
                  )}
                </div>
              </Card>

              {isStale && prediction && (
                <StaleNotice
                  predictionId={prediction.prediction_id}
                  onRerun={runPrediction}
                  canRerun={canPredict && isConnected}
                  isLoading={isPredicting}
                />
              )}

              {predictionError && (
                <Alert type="danger" code="INFERENCE_FAILED">
                  {predictionError}
                </Alert>
              )}

              <Card
                title="Maximum Stress Analysis Results"
                headerAction={
                  prediction ? (
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={handleSaveForwardCase}
                      icon={<FileText size={14} />}
                    >
                      Save Case &amp; Report
                    </Button>
                  ) : undefined
                }
              >
                <ResultsPanel prediction={prediction} isLoading={isPredicting} />
              </Card>
            </div>
          </div>
        )}

        {/* Workspace 2: Inverse Defect Inspection (Stage 5C) */}
        {currentWorkspace === 'inverse' && (
          <InverseWorkspace
            onOpenReport={(caseRecord) => {
              setCaseCount(caseHistoryService.getAllCases().length);
              setActiveReportCase(caseRecord);
            }}
          />
        )}

        {/* Workspace 3: Case History & Reports (Stage 5D) */}
        {currentWorkspace === 'history' && (
          <CaseHistoryWorkspace onOpenReport={setActiveReportCase} />
        )}
      </main>

      {/* Report Viewer Modal (Stage 5D) */}
      {activeReportCase && (
        <ReportViewer
          caseRecord={activeReportCase}
          onClose={() => setActiveReportCase(null)}
        />
      )}

      {/* Global Application Footer */}
      <Footer />
    </div>
  );
};

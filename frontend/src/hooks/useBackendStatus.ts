import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../api/client';
import { HealthResponse, ModelSummary } from '../api/types';

export function useBackendStatus() {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isReady, setIsReady] = useState<boolean>(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const checkStatus = useCallback(async () => {
    try {
      const [healthData, readinessData, modelsList] = await Promise.all([
        apiClient.getHealth(),
        apiClient.getReadiness().catch(() => ({ ready: false, service_healthy: true, all_models_loaded: false, details: {} })),
        apiClient.getModels().catch(() => []),
      ]);

      setIsConnected(true);
      setIsReady(readinessData.ready);
      setHealth(healthData);
      setModels(modelsList);
      setError(null);
    } catch (err: any) {
      setIsConnected(false);
      setIsReady(false);
      setError(err.message || 'Cannot reach SimuStruct Backend on :8000');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    checkStatus();
    // Poll health every 15 seconds to track backend connectivity
    const interval = setInterval(checkStatus, 15000);
    return () => clearInterval(interval);
  }, [checkStatus]);

  return {
    isConnected,
    isReady,
    health,
    models,
    error,
    isLoading,
    refreshStatus: checkStatus,
  };
}


/**
 * SimuStruct Centralized API Client
 * Configurable base URL, typed requests, structured error extraction, and timeout handling.
 */
import {
  HealthResponse,
  ReadinessResponse,
  ModelSummary,
  ModelDetail,
  GeometryValidationRequest,
  GeometryValidationResponse,
  ForwardPredictionRequest,
  ForwardPredictionResponse,
  InversePredictionRequest,
  InversePredictionResponse,
} from './types';

// Safe production and development API base URL resolution
const DEFAULT_PROD_API = 'https://fem-backend-ydrr.onrender.com/api/v1';

export const getBaseUrl = (): string => {
  // 1. Explicit build/runtime environment variable override
  if (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_API_URL) {
    return (import.meta as any).env.VITE_API_URL;
  }
  if (typeof globalThis !== 'undefined' && (globalThis as any).process?.env?.VITE_API_URL) {
    return (globalThis as any).process.env.VITE_API_URL;
  }
  // 2. In browser: if deployed on Vercel or any cloud domain (not local dev)
  if (typeof window !== 'undefined' && window.location) {
    const host = window.location.hostname;
    if (host !== 'localhost' && host !== '127.0.0.1' && !host.startsWith('192.168.')) {
      return DEFAULT_PROD_API;
    }
  }
  // 3. Local development proxy fallback
  return '/api/v1';
};

export const API_BASE_URL = getBaseUrl();

export class ApiError extends Error {
  public status: number;
  public code: string;
  public details?: any[];

  constructor(status: number, message: string, code = 'API_ERROR', details?: any[]) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {},
  timeoutMs = 15000
): Promise<T> {
  const url = `${API_BASE_URL.replace(/\/$/, '')}/${endpoint.replace(/^\//, '')}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
        ...(options.headers || {}),
      },
    });

    clearTimeout(timer);

    let data: any = null;
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      data = await response.json();
    } else {
      const text = await response.text();
      data = { message: text };
    }

    if (!response.ok) {
      const message = data?.detail?.message || data?.message || `HTTP Error ${response.status}`;
      const code = data?.detail?.code || data?.code || `HTTP_${response.status}`;
      const details = data?.detail?.details || data?.details || [];
      throw new ApiError(response.status, message, code, details);
    }

    return data as T;
  } catch (err: any) {
    clearTimeout(timer);
    if (err.name === 'AbortError') {
      throw new ApiError(408, 'Request timed out after 15 seconds.', 'TIMEOUT');
    }
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(0, err.message || 'Network connection failed.', 'NETWORK_ERROR');
  }
}

export const apiClient = {
  getBaseUrl: () => API_BASE_URL,

  // 1. Health & Readiness
  getHealth: () => request<HealthResponse>('/health'),
  getReadiness: () => request<ReadinessResponse>('/readiness'),

  // 2. Model Registry
  getModels: () => request<ModelSummary[]>('/models'),
  getModelDetail: (modelId: string) => request<ModelDetail>(`/models/${encodeURIComponent(modelId)}`),

  // 3. Geometry Validation
  validateGeometry: (data: GeometryValidationRequest) =>
    request<GeometryValidationResponse>('/validate/geometry', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // 4. Forward Stress Prediction
  predictForward: (data: ForwardPredictionRequest) =>
    request<ForwardPredictionResponse>('/predict/forward', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // 5. Inverse Defect Localization (Model D5 U-Net)
  predictInverse: (data: InversePredictionRequest) =>
    request<InversePredictionResponse>('/predict/inverse', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
};

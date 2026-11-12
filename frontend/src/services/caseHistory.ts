/**
 * SimuStruct Case History Service
 * Provides immutable client-side case record storage, retrieval, and cryptographic hashing.
 * Persists records to browser localStorage with full integrity metadata.
 */
import { CaseRecord, AnalysisType } from '../api/types';

const STORAGE_KEY = 'simustruct_case_history_v1';

export async function computeContentHash(data: any): Promise<string> {
  const serialized = JSON.stringify(data, Object.keys(data).sort());
  if (typeof crypto !== 'undefined' && crypto.subtle) {
    try {
      const msgUint8 = new TextEncoder().encode(serialized);
      const hashBuffer = await crypto.subtle.digest('SHA-256', msgUint8);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('').slice(0, 16);
    } catch {
      // fallback
    }
  }
  // Deterministic fallback hash for Node / non-subtle crypto
  let hash = 0;
  for (let i = 0; i < serialized.length; i++) {
    const char = serialized.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash |= 0;
  }
  return Math.abs(hash).toString(16).padStart(16, '0');
}

export function generateCaseId(type: AnalysisType): string {
  const dateStr = new Date().toISOString().slice(0, 10).replace(/-/g, '');
  const rand = Math.random().toString(36).substring(2, 6).toUpperCase();
  const prefix = type === 'FORWARD' ? 'FWD' : type === 'INVERSE' ? 'INV' : 'LNK';
  return `CASE-${prefix}-${dateStr}-${rand}`;
}

export const caseHistoryService = {
  getAllCases(): CaseRecord[] {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );
    } catch (err) {
      console.warn('Failed to load case history from localStorage:', err);
      return [];
    }
  },

  getCaseById(caseId: string): CaseRecord | undefined {
    const cases = this.getAllCases();
    return cases.find((c) => c.case_id === caseId);
  },

  async saveCase(record: Omit<CaseRecord, 'content_hash'>): Promise<CaseRecord> {
    const hash = await computeContentHash({
      req: record.forward_request || record.inverse_request,
      res: record.forward_response || record.inverse_response,
    });
    const completeRecord: CaseRecord = {
      ...record,
      content_hash: hash,
    };

    const existing = this.getAllCases();
    // Prepend new record (immutable: new run creates a new record)
    const updated = [completeRecord, ...existing.filter((c) => c.case_id !== completeRecord.case_id)];
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    } catch (err) {
      console.warn('LocalStorage quota or storage error while saving case record:', err);
    }
    return completeRecord;
  },

  deleteCase(caseId: string): void {
    const existing = this.getAllCases();
    const updated = existing.filter((c) => c.case_id !== caseId);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    } catch (err) {
      console.warn('Failed to delete case record:', err);
    }
  },

  clearAll(): void {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (err) {
      console.warn('Failed to clear case history:', err);
    }
  },
};

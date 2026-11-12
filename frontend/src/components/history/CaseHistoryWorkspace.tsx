import React, { useState, useEffect } from 'react';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';
import { CaseRecord, AnalysisType } from '../../api/types';
import { caseHistoryService } from '../../services/caseHistory';
import { exportCaseAsJson } from '../../utils/exportUtils';
import {
  History,
  Search,
  FileText,
  Download,
  Trash2,
  Columns2,
} from 'lucide-react';

interface CaseHistoryWorkspaceProps {
  onOpenReport: (record: CaseRecord) => void;
}

export const CaseHistoryWorkspace: React.FC<CaseHistoryWorkspaceProps> = ({ onOpenReport }) => {
  const [cases, setCases] = useState<CaseRecord[]>([]);
  const [filterType, setFilterType] = useState<'ALL' | AnalysisType>('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedForComparison, setSelectedForComparison] = useState<string[]>([]);
  const [isComparing, setIsComparing] = useState(false);

  const loadCases = () => {
    const list = caseHistoryService.getAllCases();
    setCases(list);
  };

  useEffect(() => {
    loadCases();
  }, []);

  const handleDelete = (caseId: string) => {
    if (window.confirm(`Delete case record ${caseId}? This action cannot be undone.`)) {
      caseHistoryService.deleteCase(caseId);
      loadCases();
      setSelectedForComparison((prev) => prev.filter((id) => id !== caseId));
    }
  };

  const handleExportAll = () => {
    const jsonStr = JSON.stringify(cases, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `simustruct_case_history_export_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const toggleComparisonSelect = (caseId: string) => {
    setSelectedForComparison((prev) => {
      if (prev.includes(caseId)) {
        return prev.filter((id) => id !== caseId);
      }
      if (prev.length >= 2) {
        return [prev[1], caseId]; // Keep only 2 for comparison
      }
      return [...prev, caseId];
    });
  };

  const filteredCases = cases.filter((c) => {
    const matchesType = filterType === 'ALL' || c.analysis_type === filterType;
    const matchesSearch =
      c.case_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      c.title.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesType && matchesSearch;
  });

  const forwardCount = cases.filter((c) => c.analysis_type === 'FORWARD').length;
  const inverseCount = cases.filter((c) => c.analysis_type === 'INVERSE').length;

  const compCaseA = cases.find((c) => c.case_id === selectedForComparison[0]);
  const compCaseB = cases.find((c) => c.case_id === selectedForComparison[1]);

  return (
    <div className="space-y-4">
      {/* Header Banner */}
      <div className="bg-[#101622] border border-border rounded-lg p-3.5 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-text-primary flex items-center gap-2">
            <History className="w-4 h-4 text-accent-cyan" />
            Engineering Case History & Traceability Audit System
          </h2>
          <p className="text-xs text-text-muted mt-0.5">
            Immutable client-side case log preserving exact inputs, outputs, timestamps, and model hashes.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={handleExportAll}
            disabled={cases.length === 0}
            icon={<Download className="w-3.5 h-3.5" />}
          >
            Export All JSON
          </Button>
          {selectedForComparison.length === 2 && (
            <Button
              variant="primary"
              size="sm"
              onClick={() => setIsComparing(true)}
              icon={<Columns2 className="w-3.5 h-3.5" />}
            >
              Compare Selected (2)
            </Button>
          )}
        </div>
      </div>

      {/* Filter and Search Bar */}
      <Card>
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
          {/* Filter Pills */}
          <div className="flex items-center gap-1.5 w-full sm:w-auto">
            <button
              type="button"
              onClick={() => setFilterType('ALL')}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                filterType === 'ALL'
                  ? 'bg-accent-cyan text-[#0a0d14] font-semibold'
                  : 'bg-surface-elevated text-text-muted hover:text-text-primary'
              }`}
            >
              All Records ({cases.length})
            </button>
            <button
              type="button"
              onClick={() => setFilterType('FORWARD')}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                filterType === 'FORWARD'
                  ? 'bg-accent-cyan text-[#0a0d14] font-semibold'
                  : 'bg-surface-elevated text-text-muted hover:text-text-primary'
              }`}
            >
              Forward ({forwardCount})
            </button>
            <button
              type="button"
              onClick={() => setFilterType('INVERSE')}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                filterType === 'INVERSE'
                  ? 'bg-accent-cyan text-[#0a0d14] font-semibold'
                  : 'bg-surface-elevated text-text-muted hover:text-text-primary'
              }`}
            >
              Inverse ({inverseCount})
            </button>
          </div>

          {/* Search Input */}
          <div className="relative w-full sm:w-64">
            <Search className="w-3.5 h-3.5 text-text-muted absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search by Case ID or title..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-[#0a0d14] border border-border rounded pl-8 pr-3 py-1.5 text-xs text-text-primary focus:border-accent-cyan outline-none"
            />
          </div>
        </div>
      </Card>

      {/* Cases Table */}
      <Card>
        {filteredCases.length === 0 ? (
          <div className="py-12 text-center text-text-muted text-xs">
            No case records found. Run an analysis in Forward or Inverse workspace and click &quot;Save to Case &amp; Report&quot;.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-border/80 text-[11px] font-semibold text-text-muted uppercase tracking-wider">
                  <th className="py-2 px-3 w-8">Comp.</th>
                  <th className="py-2 px-3">Case ID</th>
                  <th className="py-2 px-3">Type</th>
                  <th className="py-2 px-3">Date (UTC)</th>
                  <th className="py-2 px-3">Specimen (L×H)</th>
                  <th className="py-2 px-3">Key Output</th>
                  <th className="py-2 px-3">Model & Latency</th>
                  <th className="py-2 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/40 font-mono">
                {filteredCases.map((record) => {
                  const isFwd = record.analysis_type === 'FORWARD';
                  const isInv = record.analysis_type === 'INVERSE';
                  const isChecked = selectedForComparison.includes(record.case_id);

                  return (
                    <tr key={record.case_id} className="hover:bg-surface-elevated/40 transition-colors">
                      <td className="py-2.5 px-3">
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => toggleComparisonSelect(record.case_id)}
                          className="rounded border-border accent-accent-cyan cursor-pointer"
                        />
                      </td>
                      <td className="py-2.5 px-3 font-semibold text-accent-cyan">
                        <button
                          type="button"
                          onClick={() => onOpenReport(record)}
                          className="hover:underline flex items-center gap-1"
                        >
                          {record.case_id}
                        </button>
                      </td>
                      <td className="py-2.5 px-3">
                        <Badge variant={isFwd ? 'cyan' : 'green'}>{record.analysis_type}</Badge>
                      </td>
                      <td className="py-2.5 px-3 text-text-muted text-[11px]">
                        {new Date(record.created_at).toLocaleDateString()} {new Date(record.created_at).toLocaleTimeString()}
                      </td>
                      <td className="py-2.5 px-3 text-text-secondary">
                        {isFwd && record.forward_request
                          ? `${record.forward_request.plate_length_mm}×${record.forward_request.plate_height_mm}`
                          : isInv && record.inverse_request
                          ? `${record.inverse_request.plate_length_mm}×${record.inverse_request.plate_height_mm}`
                          : '1000×1000'}
                      </td>
                      <td className="py-2.5 px-3 font-bold">
                        {isFwd && record.forward_response ? (
                          <span className="text-accent-cyan">
                            σ_max: {record.forward_response.predicted_max_stress_psi.toFixed(1)} psi (Kt={record.forward_response.predicted_Kt_element.toFixed(2)})
                          </span>
                        ) : isInv && record.inverse_response ? (
                          <span className="text-emerald-400">
                            ({record.inverse_response.predicted_defect_x_mm.toFixed(1)},{' '}
                            {record.inverse_response.predicted_defect_y_mm.toFixed(1)}) mm (±
                            {record.inverse_response.uncertainty_radius_mm.toFixed(1)})
                          </span>
                        ) : (
                          '-'
                        )}
                      </td>
                      <td className="py-2.5 px-3 text-[11px] text-text-muted">
                        {isFwd && record.forward_response
                          ? `${record.forward_response.model_id} (${record.forward_response.execution_time_ms.toFixed(0)}ms)`
                          : isInv && record.inverse_response
                          ? `${record.inverse_response.model_id} (${record.inverse_response.execution_time_ms.toFixed(0)}ms)`
                          : '-'}
                      </td>
                      <td className="py-2.5 px-3 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          <button
                            type="button"
                            onClick={() => onOpenReport(record)}
                            title="View Full Report"
                            className="p-1 rounded hover:bg-border/60 text-text-muted hover:text-text-primary"
                          >
                            <FileText className="w-3.5 h-3.5 text-accent-cyan" />
                          </button>
                          <button
                            type="button"
                            onClick={() => exportCaseAsJson(record)}
                            title="Export JSON"
                            className="p-1 rounded hover:bg-border/60 text-text-muted hover:text-text-primary"
                          >
                            <Download className="w-3.5 h-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => handleDelete(record.case_id)}
                            title="Delete Record"
                            className="p-1 rounded hover:bg-border/60 text-text-muted hover:text-red-400"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* Side-by-Side Comparison Modal */}
      {isComparing && compCaseA && compCaseB && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex justify-center items-center p-4">
          <div className="w-full max-w-4xl bg-[#0f141f] border border-border rounded-xl p-6 space-y-4 text-xs font-mono">
            <div className="flex items-center justify-between pb-3 border-b border-border">
              <h3 className="text-sm font-bold text-text-primary flex items-center gap-2">
                <Columns2 className="w-4 h-4 text-accent-cyan" />
                Engineering Case Comparison: {compCaseA.case_id} vs {compCaseB.case_id}
              </h3>
              <Button variant="ghost" size="sm" onClick={() => setIsComparing(false)}>
                Close Comparison
              </Button>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {/* Case A */}
              <div className="p-3 rounded bg-surface-elevated/40 border border-border">
                <div className="font-bold text-accent-cyan mb-2 text-sm">{compCaseA.case_id}</div>
                <div>Type: {compCaseA.analysis_type}</div>
                <div>Date: {new Date(compCaseA.created_at).toLocaleString()}</div>
                <div>Hash: {compCaseA.content_hash}</div>
                <div className="mt-3 pt-2 border-t border-border/40">
                  {compCaseA.forward_response && (
                    <div>
                      <div>σ_max: {compCaseA.forward_response.predicted_max_stress_psi} psi</div>
                      <div>Kt: {compCaseA.forward_response.predicted_Kt_element}</div>
                      <div>Uncertainty: ±{compCaseA.forward_response.ensemble_uncertainty_psi} psi</div>
                    </div>
                  )}
                  {compCaseA.inverse_response && (
                    <div>
                      <div>
                        Defect: ({compCaseA.inverse_response.predicted_defect_x_mm},{' '}
                        {compCaseA.inverse_response.predicted_defect_y_mm}) mm
                      </div>
                      <div>Dispersion: ±{compCaseA.inverse_response.uncertainty_radius_mm} mm</div>
                      <div>Status: {compCaseA.inverse_response.abstention.status}</div>
                    </div>
                  )}
                </div>
              </div>

              {/* Case B */}
              <div className="p-3 rounded bg-surface-elevated/40 border border-border">
                <div className="font-bold text-emerald-400 mb-2 text-sm">{compCaseB.case_id}</div>
                <div>Type: {compCaseB.analysis_type}</div>
                <div>Date: {new Date(compCaseB.created_at).toLocaleString()}</div>
                <div>Hash: {compCaseB.content_hash}</div>
                <div className="mt-3 pt-2 border-t border-border/40">
                  {compCaseB.forward_response && (
                    <div>
                      <div>σ_max: {compCaseB.forward_response.predicted_max_stress_psi} psi</div>
                      <div>Kt: {compCaseB.forward_response.predicted_Kt_element}</div>
                      <div>Uncertainty: ±{compCaseB.forward_response.ensemble_uncertainty_psi} psi</div>
                    </div>
                  )}
                  {compCaseB.inverse_response && (
                    <div>
                      <div>
                        Defect: ({compCaseB.inverse_response.predicted_defect_x_mm},{' '}
                        {compCaseB.inverse_response.predicted_defect_y_mm}) mm
                      </div>
                      <div>Dispersion: ±{compCaseB.inverse_response.uncertainty_radius_mm} mm</div>
                      <div>Status: {compCaseB.inverse_response.abstention.status}</div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * webview.ts
 * ==========
 * The sidebar UI panel that displays review results.
 * Uses HTML + CSS rendered in a VS Code WebviewView.
 */

import * as vscode from 'vscode';
import { ReviewResponse } from './reviewer';

export class ReviewResultsProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = 'aiReviewer.resultsPanel';
  private _view?: vscode.WebviewView;

  constructor(private readonly _extensionUri: vscode.Uri) {}

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this._view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._extensionUri],
    };

    webviewView.webview.html = this.getWelcomeHtml();
  }

  /** Show loading state */
  showLoading(filename: string) {
    if (this._view) {
      this._view.webview.html = this.getLoadingHtml(filename);
      this._view.show(true);
    }
  }

  /** Show review results */
  showResults(response: ReviewResponse, filename: string) {
    if (this._view) {
      this._view.webview.html = this.getResultsHtml(response, filename);
      this._view.show(true);
    }
  }

  /** Show error state */
  showError(message: string) {
    if (this._view) {
      this._view.webview.html = this.getErrorHtml(message);
    }
  }

  // ─── HTML Generators ───────────────────────────────────────────────────────

  private getScoreColor(score: number): string {
    if (score >= 80) return '#4caf50';
    if (score >= 60) return '#ff9800';
    if (score >= 40) return '#f44336';
    return '#b71c1c';
  }

  private getSeverityColor(severity: string): string {
    const colors: Record<string, string> = {
      critical: '#b71c1c',
      high: '#f44336',
      medium: '#ff9800',
      low: '#ffc107',
      info: '#2196f3',
    };
    return colors[severity] || '#757575';
  }

  private getSharedStyles(): string {
    return `
      <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          font-family: var(--vscode-font-family);
          font-size: var(--vscode-font-size);
          color: var(--vscode-foreground);
          background: var(--vscode-sideBar-background);
          padding: 12px;
          line-height: 1.5;
        }
        .section {
          margin-bottom: 16px;
          background: var(--vscode-editor-background);
          border-radius: 6px;
          overflow: hidden;
          border: 1px solid var(--vscode-panel-border);
        }
        .section-header {
          padding: 8px 12px;
          font-weight: 600;
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          background: var(--vscode-titleBar-activeBackground);
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .section-body { padding: 10px 12px; }
        .badge {
          display: inline-block;
          padding: 1px 6px;
          border-radius: 10px;
          font-size: 10px;
          font-weight: 700;
          color: white;
          text-transform: uppercase;
        }
        .issue-item {
          padding: 8px 0;
          border-bottom: 1px solid var(--vscode-panel-border);
        }
        .issue-item:last-child { border-bottom: none; }
        .issue-title {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 4px;
        }
        .issue-desc { font-size: 12px; color: var(--vscode-descriptionForeground); }
        .issue-fix {
          margin-top: 4px;
          font-size: 11px;
          padding: 4px 8px;
          background: var(--vscode-textBlockQuote-background);
          border-left: 2px solid var(--vscode-textLink-foreground);
          border-radius: 2px;
        }
        .score-circle {
          width: 80px; height: 80px;
          border-radius: 50%;
          display: flex; flex-direction: column;
          align-items: center; justify-content: center;
          margin: 0 auto 12px;
          border: 3px solid;
        }
        .score-num { font-size: 28px; font-weight: 700; line-height: 1; }
        .score-grade { font-size: 14px; font-weight: 600; }
        .bar-container { margin-bottom: 6px; }
        .bar-label {
          display: flex; justify-content: space-between;
          font-size: 11px; margin-bottom: 2px;
        }
        .bar-track {
          height: 4px; background: var(--vscode-panel-border);
          border-radius: 2px; overflow: hidden;
        }
        .bar-fill { height: 100%; border-radius: 2px; transition: width 0.5s; }
        code {
          font-family: var(--vscode-editor-font-family);
          font-size: 11px;
          background: var(--vscode-textCodeBlock-background);
          padding: 1px 4px;
          border-radius: 3px;
        }
        pre {
          font-family: var(--vscode-editor-font-family);
          font-size: 11px;
          background: var(--vscode-textCodeBlock-background);
          padding: 8px;
          border-radius: 4px;
          overflow-x: auto;
          margin-top: 6px;
          white-space: pre-wrap;
        }
        .empty-state {
          text-align: center;
          color: var(--vscode-descriptionForeground);
          font-size: 12px;
          padding: 12px 0;
        }
        .tag {
          display: inline-block;
          padding: 1px 8px;
          background: var(--vscode-badge-background);
          color: var(--vscode-badge-foreground);
          border-radius: 10px;
          font-size: 11px;
        }
      </style>
    `;
  }

  private getWelcomeHtml(): string {
    return `<!DOCTYPE html><html><head>${this.getSharedStyles()}</head><body>
      <div style="text-align:center; padding: 32px 16px;">
        <div style="font-size:40px; margin-bottom:12px;">🤖</div>
        <div style="font-weight:600; margin-bottom:8px;">AI Code Reviewer</div>
        <div style="font-size:12px; color:var(--vscode-descriptionForeground); margin-bottom:16px);">
          Right-click your code and select<br><strong>"AI: Review This File"</strong>
        </div>
        <div style="font-size:11px; color:var(--vscode-descriptionForeground);">
          Keyboard: <code>Ctrl+Shift+R</code>
        </div>
      </div>
    </body></html>`;
  }

  private getLoadingHtml(filename: string): string {
    return `<!DOCTYPE html><html><head>${this.getSharedStyles()}
      <style>
        @keyframes spin { to { transform: rotate(360deg); } }
        .spinner {
          width: 32px; height: 32px;
          border: 3px solid var(--vscode-panel-border);
          border-top-color: var(--vscode-progressBar-background);
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
          margin: 0 auto 12px;
        }
        @keyframes pulse { 0%,100% { opacity: 0.4; } 50% { opacity: 1; } }
        .dot { animation: pulse 1.2s ease-in-out infinite; }
        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }
      </style>
    </head><body>
      <div style="text-align:center; padding: 32px 16px;">
        <div class="spinner"></div>
        <div style="font-weight:600; margin-bottom:4px;">Analyzing code<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>
        <div style="font-size:11px; color:var(--vscode-descriptionForeground);">${filename}</div>
      </div>
    </body></html>`;
  }

  private getResultsHtml(r: ReviewResponse, filename: string): string {
    const scoreColor = this.getScoreColor(r.score);
    
    // Build bugs section
    const bugsHtml = r.bugs.length === 0
      ? '<div class="empty-state">✅ No bugs detected!</div>'
      : r.bugs.map(bug => `
          <div class="issue-item">
            <div class="issue-title">
              <span class="badge" style="background:${this.getSeverityColor(bug.severity)}">${bug.severity}</span>
              ${bug.line ? `<code>Line ${bug.line}</code>` : ''}
            </div>
            <div class="issue-desc">${bug.description}</div>
            ${bug.suggestion ? `<div class="issue-fix">💡 ${bug.suggestion}</div>` : ''}
          </div>
        `).join('');

    // Build security section
    const securityHtml = r.security_issues.length === 0
      ? '<div class="empty-state">🔒 No security issues found!</div>'
      : r.security_issues.map(issue => `
          <div class="issue-item">
            <div class="issue-title">
              <span class="badge" style="background:${this.getSeverityColor(issue.severity)}">${issue.severity}</span>
              <strong>${issue.issue_type}</strong>
              ${issue.line ? `<code>Line ${issue.line}</code>` : ''}
            </div>
            <div class="issue-desc">${issue.description}</div>
            ${issue.fix ? `<div class="issue-fix">🛠️ ${issue.fix}</div>` : ''}
          </div>
        `).join('');

    // Build improvements section
    const improvementsHtml = r.improvements.length === 0
      ? '<div class="empty-state">⭐ Code looks great!</div>'
      : r.improvements.map(imp => `
          <div class="issue-item">
            <div class="issue-title">
              <span class="tag">${imp.category}</span>
              ${imp.line ? `<code>Line ${imp.line}</code>` : ''}
            </div>
            <div class="issue-desc">${imp.description}</div>
            ${imp.code_example ? `<pre>${imp.code_example}</pre>` : ''}
          </div>
        `).join('');

    // Build breakdown bars
    const breakdown = r.breakdown;
    const bars = Object.entries(breakdown).map(([key, val]) => `
      <div class="bar-container">
        <div class="bar-label">
          <span>${key.charAt(0).toUpperCase() + key.slice(1)}</span>
          <span>${val}/100</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${val}%; background:${this.getScoreColor(val)};"></div>
        </div>
      </div>
    `).join('');

    return `<!DOCTYPE html><html><head>${this.getSharedStyles()}</head><body>

      <!-- Score Section -->
      <div class="section">
        <div class="section-header">📊 Quality Score</div>
        <div class="section-body">
          <div class="score-circle" style="border-color:${scoreColor}; color:${scoreColor}">
            <div class="score-num">${r.score}</div>
            <div class="score-grade">${r.grade}</div>
          </div>
          <div style="font-size:11px; text-align:center; margin-bottom:12px; color:var(--vscode-descriptionForeground);">
            ${filename} · ${r.language_detected} · ${r.review_time_ms}ms
          </div>
          ${bars}
        </div>
      </div>

      <!-- Summary -->
      <div class="section">
        <div class="section-header">💬 Summary</div>
        <div class="section-body" style="font-size:12px;">${r.summary}</div>
      </div>

      <!-- Bugs -->
      <div class="section">
        <div class="section-header">
          🐛 Bugs 
          ${r.bugs.length > 0 ? `<span class="badge" style="background:#f44336">${r.bugs.length}</span>` : ''}
        </div>
        <div class="section-body">${bugsHtml}</div>
      </div>

      <!-- Security -->
      <div class="section">
        <div class="section-header">
          🔒 Security
          ${r.security_issues.length > 0 ? `<span class="badge" style="background:#b71c1c">${r.security_issues.length}</span>` : ''}
        </div>
        <div class="section-body">${securityHtml}</div>
      </div>

      <!-- Improvements -->
      <div class="section">
        <div class="section-header">
          💡 Improvements
          ${r.improvements.length > 0 ? `<span class="badge" style="background:#1976d2">${r.improvements.length}</span>` : ''}
        </div>
        <div class="section-body">${improvementsHtml}</div>
      </div>

      <div style="font-size:10px; color:var(--vscode-descriptionForeground); text-align:center; padding:8px 0;">
        Powered by ${r.model_used} model
      </div>

    </body></html>`;
  }

  private getErrorHtml(message: string): string {
    return `<!DOCTYPE html><html><head>${this.getSharedStyles()}</head><body>
      <div style="text-align:center; padding: 24px 16px;">
        <div style="font-size:32px; margin-bottom:8px;">❌</div>
        <div style="font-weight:600; margin-bottom:8px; color:#f44336;">Review Failed</div>
        <div style="font-size:11px; color:var(--vscode-descriptionForeground); text-align:left;
                    background:var(--vscode-textCodeBlock-background); padding:10px; border-radius:4px;">
          ${message.replace(/\n/g, '<br>')}
        </div>
        <div style="font-size:11px; margin-top:12px; color:var(--vscode-descriptionForeground);">
          Make sure the backend is running:<br>
          <code>cd backend && uvicorn app.main:app --reload</code>
        </div>
      </div>
    </body></html>`;
  }
}

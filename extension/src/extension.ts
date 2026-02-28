/**
 * extension.ts
 * ============
 * Main entry point for the VS Code extension.
 * This file is what VS Code loads when the extension activates.
 * 
 * It registers all commands, sets up diagnostics (red squiggles),
 * and coordinates between the API client and the sidebar UI.
 */

import * as vscode from 'vscode';
import { ReviewerApiClient, ReviewResponse, Bug, SecurityIssue } from './reviewer';
import { ReviewResultsProvider } from './webview';

// ─── Extension State ───────────────────────────────────────────────────────────

let diagnosticsCollection: vscode.DiagnosticCollection;
let apiClient: ReviewerApiClient;
let resultsProvider: ReviewResultsProvider;
let statusBarItem: vscode.StatusBarItem;
let lastReview: ReviewResponse | null = null;

// ─── ACTIVATE ─────────────────────────────────────────────────────────────────
// This runs once when the extension first loads

export function activate(context: vscode.ExtensionContext) {
  console.log('AI Code Reviewer activated!');

  // Initialize components
  apiClient = new ReviewerApiClient();
  diagnosticsCollection = vscode.languages.createDiagnosticCollection('aiReviewer');
  resultsProvider = new ReviewResultsProvider(context.extensionUri);

  // Status bar item (shows in bottom bar)
  statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  statusBarItem.command = 'aiReviewer.reviewFile';
  statusBarItem.text = '$(robot) Review';
  statusBarItem.tooltip = 'AI: Review This File (Ctrl+Shift+R)';
  statusBarItem.show();

  // Register the sidebar webview
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      ReviewResultsProvider.viewType,
      resultsProvider
    )
  );

  // ─── Register Commands ────────────────────────────────────────────────────

  // Command 1: Review full file
  context.subscriptions.push(
    vscode.commands.registerCommand('aiReviewer.reviewFile', reviewCurrentFile)
  );

  // Command 2: Review selected text only
  context.subscriptions.push(
    vscode.commands.registerCommand('aiReviewer.reviewSelection', reviewSelection)
  );

  // Command 3: Quick security scan
  context.subscriptions.push(
    vscode.commands.registerCommand('aiReviewer.quickScan', quickScan)
  );

  // Command 4: Clear all diagnostics
  context.subscriptions.push(
    vscode.commands.registerCommand('aiReviewer.clearDiagnostics', () => {
      diagnosticsCollection.clear();
      lastReview = null;
      statusBarItem.text = '$(robot) Review';
      statusBarItem.backgroundColor = undefined;
    })
  );

  // ─── Auto-review on save (optional) ──────────────────────────────────────
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument(async (document) => {
      const config = vscode.workspace.getConfiguration('aiReviewer');
      if (config.get<boolean>('reviewOnSave')) {
        await performReview(document, false); // silent mode
      }
    })
  );

  // ─── Clean up on deactivate ───────────────────────────────────────────────
  context.subscriptions.push(diagnosticsCollection);
  context.subscriptions.push(statusBarItem);

  // Check backend health on startup
  checkBackendHealth();
}

// ─── COMMAND HANDLERS ─────────────────────────────────────────────────────────

async function reviewCurrentFile() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage('No active file to review');
    return;
  }
  await performReview(editor.document, true);
}

async function reviewSelection() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage('No active editor');
    return;
  }

  const selection = editor.selection;
  if (selection.isEmpty) {
    vscode.window.showWarningMessage('Please select some code first');
    return;
  }

  const selectedText = editor.document.getText(selection);
  const filename = `selection from ${editor.document.fileName.split('/').pop()}`;
  const language = getLanguage(editor.document);

  await performReviewWithCode(selectedText, language, filename);
}

async function quickScan() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) { return; }

  const code = editor.document.getText();
  const language = getLanguage(editor.document);

  statusBarItem.text = '$(sync~spin) Scanning...';

  try {
    const result = await apiClient.quickScan(code, language);
    const issues = [
      ...(result.critical_issues || []),
      ...(result.security_issues || [])
    ].length;

    if (issues === 0) {
      vscode.window.showInformationMessage('✅ Quick scan: No critical issues found!');
    } else {
      vscode.window.showWarningMessage(
        `🔒 Quick scan: ${issues} issue(s) found. Run full review for details.`,
        'Review Now'
      ).then(action => {
        if (action === 'Review Now') {
          reviewCurrentFile();
        }
      });
    }
  } catch (err: any) {
    vscode.window.showErrorMessage(`Quick scan failed: ${err.message}`);
  } finally {
    statusBarItem.text = '$(robot) Review';
  }
}

// ─── CORE REVIEW LOGIC ────────────────────────────────────────────────────────

async function performReview(document: vscode.TextDocument, showProgress: boolean) {
  const code = document.getText();
  const language = getLanguage(document);
  const filename = document.fileName.split('\\').pop()?.split('/').pop() || 'file';

  await performReviewWithCode(code, language, filename, document);
}

async function performReviewWithCode(
  code: string,
  language: string,
  filename: string,
  document?: vscode.TextDocument
) {
  // Show loading state
  resultsProvider.showLoading(filename);
  statusBarItem.text = '$(sync~spin) Reviewing...';
  statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');

  try {
    // Call the API
    const review = await apiClient.reviewCode(code, language, filename);
    lastReview = review;

    // Update sidebar
    resultsProvider.showResults(review, filename);

    // Add diagnostic squiggles to the editor
    if (document) {
      applyDiagnostics(document, review);
    }

    // Update status bar
    const color = review.score >= 70
      ? undefined
      : new vscode.ThemeColor('statusBarItem.errorBackground');
    
    statusBarItem.text = `$(robot) ${review.score}/100 ${review.grade}`;
    statusBarItem.backgroundColor = color;

    // Show summary notification
    const config = vscode.workspace.getConfiguration('aiReviewer');
    const minScore = config.get<number>('minScoreToWarn') || 60;

    if (review.score < minScore) {
      const action = await vscode.window.showWarningMessage(
        `⚠️ Code scored ${review.score}/100. ${review.bugs.length} bug(s), ` +
        `${review.security_issues.length} security issue(s) found.`,
        'View Details'
      );
      if (action === 'View Details') {
        vscode.commands.executeCommand('aiReviewer.resultsPanel.focus');
      }
    } else if (review.score >= 90) {
      vscode.window.showInformationMessage(
        `✅ Excellent code! Scored ${review.score}/100 (Grade: ${review.grade})`
      );
    }

  } catch (err: any) {
    resultsProvider.showError(err.message);
    statusBarItem.text = '$(robot) Review Failed';
    statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
    vscode.window.showErrorMessage(`Review failed: ${err.message}`);
  }
}

// ─── DIAGNOSTICS (RED/YELLOW SQUIGGLES) ──────────────────────────────────────

function applyDiagnostics(document: vscode.TextDocument, review: ReviewResponse) {
  const diagnostics: vscode.Diagnostic[] = [];
  const config = vscode.workspace.getConfiguration('aiReviewer');
  
  if (!config.get<boolean>('showInlineDecorations', true)) {
    return;
  }

  // Add bug diagnostics
  for (const bug of review.bugs) {
    if (bug.line === null) { continue; }
    
    const lineIndex = Math.max(0, bug.line - 1);
    if (lineIndex >= document.lineCount) { continue; }

    const line = document.lineAt(lineIndex);
    const range = new vscode.Range(
      lineIndex, line.firstNonWhitespaceCharacterIndex,
      lineIndex, line.text.length
    );

    const severity = bug.severity === 'critical' || bug.severity === 'high'
      ? vscode.DiagnosticSeverity.Error
      : vscode.DiagnosticSeverity.Warning;

    const diag = new vscode.Diagnostic(
      range,
      `🐛 ${bug.description}${bug.suggestion ? `\n💡 Fix: ${bug.suggestion}` : ''}`,
      severity
    );
    diag.source = 'AI Code Reviewer';
    diag.code = bug.severity;
    diagnostics.push(diag);
  }

  // Add security issue diagnostics
  for (const issue of review.security_issues) {
    if (issue.line === null) { continue; }
    
    const lineIndex = Math.max(0, issue.line - 1);
    if (lineIndex >= document.lineCount) { continue; }

    const line = document.lineAt(lineIndex);
    const range = new vscode.Range(
      lineIndex, 0,
      lineIndex, line.text.length
    );

    const diag = new vscode.Diagnostic(
      range,
      `🔒 ${issue.issue_type}: ${issue.description}${issue.fix ? `\n🛠️ Fix: ${issue.fix}` : ''}`,
      vscode.DiagnosticSeverity.Error
    );
    diag.source = 'AI Code Reviewer';
    diag.code = `security:${issue.issue_type}`;
    diagnostics.push(diag);
  }

  // Apply all diagnostics at once
  diagnosticsCollection.set(document.uri, diagnostics);
}

// ─── HELPERS ──────────────────────────────────────────────────────────────────

function getLanguage(document: vscode.TextDocument): string {
  // VS Code language IDs map to our backend language names
  const langMap: Record<string, string> = {
    python: 'python',
    javascript: 'javascript',
    typescript: 'typescript',
    java: 'java',
    cpp: 'cpp',
    c: 'c',
    go: 'go',
    rust: 'rust',
  };
  return langMap[document.languageId] || document.languageId;
}

async function checkBackendHealth() {
  try {
    const health = await apiClient.checkHealth();
    if (health.status === 'healthy') {
      console.log(`AI Code Reviewer backend ready (${health.model_backend} mode)`);
    }
  } catch {
    // Backend not running — show gentle warning
    vscode.window.showInformationMessage(
      'AI Code Reviewer: Backend not detected. Start it with: cd backend && uvicorn app.main:app --reload',
      'Dismiss'
    );
  }
}

// ─── DEACTIVATE ───────────────────────────────────────────────────────────────

export function deactivate() {
  diagnosticsCollection.dispose();
  statusBarItem.dispose();
}

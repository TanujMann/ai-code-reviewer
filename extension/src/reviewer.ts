import * as vscode from 'vscode';
import * as https from 'https';
import * as http from 'http';

export interface Bug {
  line: number | null;
  description: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  suggestion: string | null;
}

export interface SecurityIssue {
  line: number | null;
  issue_type: string;
  description: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  fix: string | null;
}

export interface Improvement {
  line: number | null;
  category: string;
  description: string;
  code_example: string | null;
}

export interface QualityBreakdown {
  correctness: number;
  security: number;
  performance: number;
  readability: number;
  maintainability: number;
}

export interface ReviewResponse {
  score: number;
  grade: string;
  breakdown: QualityBreakdown;
  bugs: Bug[];
  security_issues: SecurityIssue[];
  critical_issues?: Bug[];
  improvements: Improvement[];
  summary: string;
  language_detected: string;
  review_time_ms: number;
  model_used: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_backend: string;
  version: string;
}

export class ReviewerApiClient {
  private baseUrl: string;

  constructor() {
    const config = vscode.workspace.getConfiguration('aiReviewer');
    this.baseUrl = config.get<string>('apiUrl') || 'http://localhost:8000/api/v1';
  }

  private async request<T>(method: string, path: string, body?: object): Promise<T> {
    return new Promise((resolve, reject) => {
      const url = new URL(this.baseUrl + path);
      const isHttps = url.protocol === 'https:';
      const lib = isHttps ? https : http;
      const bodyStr = body ? JSON.stringify(body) : '';

      const options: http.RequestOptions = {
        hostname: url.hostname,
        port: url.port || (isHttps ? 443 : 80),
        path: url.pathname + url.search,
        method,
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(bodyStr),
        },
        timeout: 30000,
      };

      const req = lib.request(options, (res) => {
        let data = '';
        res.on('data', (chunk) => (data += chunk));
        res.on('end', () => {
          if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
            try {
              resolve(JSON.parse(data) as T);
            } catch {
              reject(new Error(`Invalid JSON: ${data.slice(0, 100)}`));
            }
          } else {
            reject(new Error(`API error ${res.statusCode}: ${data}`));
          }
        });
      });

      req.on('error', (err: any) => {
        if (err.message.includes('ECONNREFUSED')) {
          reject(new Error(`Cannot connect to backend at ${this.baseUrl}`));
        } else {
          reject(err);
        }
      });

      req.on('timeout', () => { req.destroy(); reject(new Error('Timeout')); });
      if (bodyStr) { req.write(bodyStr); }
      req.end();
    });
  }

  async checkHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('GET', '/health');
  }

  async reviewCode(code: string, language: string, filename?: string): Promise<ReviewResponse> {
    return this.request<ReviewResponse>('POST', '/review', { code, language, filename });
  }

  async quickScan(code: string, language: string): Promise<Partial<ReviewResponse>> {
    return this.request<Partial<ReviewResponse>>('POST', '/quick-scan', { code, language });
  }
}
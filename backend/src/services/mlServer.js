import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

let mlProcess = null;

function resolvePythonCommand() {
  if (process.env.ML_PYTHON_BIN?.trim()) {
    return process.env.ML_PYTHON_BIN.trim();
  }
  if (process.platform === 'win32') {
    return 'python';
  }
  return 'python3';
}

function isEmbeddedMlEnabled() {
  const flag = (process.env.ML_EMBEDDED_API ?? 'true').toLowerCase();
  return flag !== 'false' && flag !== '0' && flag !== 'no';
}

export function startEmbeddedMlApi() {
  if (!isEmbeddedMlEnabled()) {
    console.log('[ML] Embedded ML API disabled by ML_EMBEDDED_API');
    return null;
  }

  if (mlProcess) {
    return mlProcess;
  }

  const pythonCmd = resolvePythonCommand();
  const mlPort = String(process.env.ML_INTERNAL_PORT || process.env.ML_API_PORT || 8000);
  const workspaceRoot = path.resolve(__dirname, '../../..');
  const mlApiDir = path.join(workspaceRoot, 'ml_api');

  const args = ['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', mlPort];

  mlProcess = spawn(pythonCmd, args, {
    cwd: mlApiDir,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      PYTHONIOENCODING: 'utf-8',
    },
  });

  mlProcess.stdout?.on('data', (chunk) => {
    const text = String(chunk).trim();
    if (text) {
      console.log(`[ML] ${text}`);
    }
  });

  mlProcess.stderr?.on('data', (chunk) => {
    const text = String(chunk).trim();
    if (text) {
      console.error(`[ML] ${text}`);
    }
  });

  mlProcess.on('error', (error) => {
    console.error(`[ML] Failed to start embedded ML API: ${error.message}`);
  });

  mlProcess.on('exit', (code, signal) => {
    console.log(`[ML] Embedded ML API exited (code=${code ?? 'null'}, signal=${signal ?? 'null'})`);
    mlProcess = null;
  });

  console.log(`[ML] Starting embedded ML API on 127.0.0.1:${mlPort}`);
  return mlProcess;
}

export function stopEmbeddedMlApi() {
  if (!mlProcess) {
    return;
  }
  try {
    mlProcess.kill('SIGTERM');
  } catch (error) {
    console.error(`[ML] Failed to stop embedded ML API gracefully: ${error.message}`);
  }
}

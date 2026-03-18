import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import app from './app.js';
import { connectDB } from './config/db.js';
import http from 'http';
import { initSocket } from './services/socket.js';
import { startEmbeddedMlApi, stopEmbeddedMlApi } from './services/mlServer.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Load backend service configuration from backend/.env.
dotenv.config({ path: path.resolve(__dirname, '../.env') });

const port = Number(process.env.PORT || 5000);

const server = http.createServer(app);
initSocket(server);

const gracefulShutdown = (signal) => {
  console.log(`Received ${signal}. Shutting down services...`);
  stopEmbeddedMlApi();
  server.close(() => process.exit(0));
};

process.on('SIGINT', () => gracefulShutdown('SIGINT'));
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));

server.on('error', (error) => {
  if (error?.code === 'EADDRINUSE') {
    console.error(`Port ${port} is already in use. Stop the existing process or change PORT in backend/.env.`);
  } else {
    console.error('Server failed to start:', error.message);
  }

  stopEmbeddedMlApi();
  process.exit(1);
});

connectDB().finally(() => {
  server.listen(port, () => {
    console.log(`CyberShield backend running on port ${port}`);
    startEmbeddedMlApi();
  });
});

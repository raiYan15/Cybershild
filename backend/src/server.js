import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import app from './app.js';
import { connectDB } from './config/db.js';
import http from 'http';
import { initSocket } from './services/socket.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Load backend service configuration from backend/.env.
dotenv.config({ path: path.resolve(__dirname, '../.env') });

const port = Number(process.env.PORT || 5000);

const server = http.createServer(app);
initSocket(server);

connectDB().finally(() => {
  server.listen(port, () => {
    console.log(`CyberShield backend running on port ${port}`);
  });
});

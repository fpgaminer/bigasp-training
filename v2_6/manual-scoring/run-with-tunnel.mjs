import 'dotenv/config';
import { spawn } from 'node:child_process';

const { PORT, CF_TUNNEL_TOKEN } = process.env;

// Use vite without command line arguments since they're now in vite.config.ts
const vite = spawn('vite', ["--debug"], { stdio: 'inherit' });
const cf   = spawn('cloudflared',
                   ['tunnel', 'run', '--token', CF_TUNNEL_TOKEN],
                   { stdio: 'inherit' });

process.on('SIGINT', () => { vite.kill(); cf.kill(); });

import { defineConfig, devices } from '@playwright/test';

const PORT = process.env.GTO_TRAINER_PORT ?? '8082';
const HOST = process.env.GTO_TRAINER_HOST ?? '127.0.0.1';
const BASE_URL = `http://${HOST}:${PORT}`;

export default defineConfig({
  testDir: './ui-tests',
  timeout: 60_000,
  fullyParallel: true,
  retries: process.env.CI ? 2 : 0,
  use: {
    baseURL: BASE_URL,
    trace: 'on-first-retry',
  },
  webServer: {
    command: `./target/release/gto-trainer serve --addr ${HOST}:${PORT}`,
    url: BASE_URL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});

import { test, expect } from '@playwright/test';

test.describe('Web UI smoke', () => {
  test('loads landing page and displays session data', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'GTO Trainer' })).toBeVisible();
    await expect(page.getByText('Heads-up no-limit hold')).toBeVisible();
  });
});

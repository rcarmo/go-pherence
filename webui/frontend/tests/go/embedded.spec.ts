import { test, expect } from '@playwright/test';

test('QEV playground submits a bounded decision request', async ({ page }) => {
	const errors: string[] = [];
	page.on('pageerror', (e) => errors.push(e.message));
	await page.goto('/qev');
	await expect(page.getByRole('heading', { name: 'QEV Decision Playground' })).toBeVisible();
	await expect(page.locator('#status')).toContainText('ready');
	await page.getByRole('button', { name: 'Run decision' }).click();
	await expect(page.locator('.decision').first()).toContainText('"urgent": true');
	await expect(page.locator('.prob').first()).toHaveText('87.50%');
	await expect(page.locator('.pill').first()).toContainText('total: 3.50 ms');
	await page.screenshot({ path: test.info().outputPath('qev.png'), fullPage: true });
	expect(errors).toEqual([]);
});

test('embedded upstream chat streams, persists, renders errors and fits mobile', async ({
	page
}) => {
	const errors: string[] = [];
	page.on('pageerror', (e) => errors.push(e.message));
	await page.goto('/');
	const brand = page.getByRole('link', { name: 'go-pherence', exact: true });
	await expect(brand).toBeVisible();
	await expect(brand.locator('img')).toHaveAttribute('src', '/favicon.svg');
	await expect
		.poll(() => brand.locator('img').evaluate((img: HTMLImageElement) => img.naturalWidth))
		.toBe(64);
	const favicon = await page.request.get('/favicon.svg');
	expect(favicon.status()).toBe(200);
	expect(await favicon.text()).toContain('<title>go-pherence</title>');
	await expect(page.locator('textarea').first()).toBeVisible();
	await expect(page.getByRole('button', { name: 'fixture-model', exact: true })).toBeVisible();
	const response = page.waitForResponse((r) => r.url().includes('/webui/v1/chat/completions'));
	await page.locator('textarea').first().fill('Hello from the browser');
	await page.getByRole('button', { name: 'Send', exact: true }).click();
	expect((await response).status()).toBe(200);
	await expect(page.getByText('Synthetic reply from Go.', { exact: true })).toBeVisible();
	// Rendering a chunk precedes async IndexedDB persistence. Wait for the
	// durable record, not an arbitrary sleep, before testing a hard reload.
	await expect
		.poll(() =>
			page.evaluate(async () => {
				return new Promise<boolean>((resolve, reject) => {
					const request = indexedDB.open('LlamaUi');
					request.onerror = () => reject(request.error);
					request.onsuccess = () => {
						const db = request.result;
						const rows = db.transaction('messages').objectStore('messages').getAll();
						rows.onsuccess = () => {
							resolve(rows.result.some((row) => row.content === 'Synthetic reply from Go.'));
							db.close();
						};
						rows.onerror = () => {
							db.close();
							reject(rows.error);
						};
					};
				});
			})
		)
		.toBe(true);
	await page.reload();
	await expect(page.getByText('Synthetic reply from Go.', { exact: true })).toBeVisible();
	await page.screenshot({ path: test.info().outputPath('desktop.png'), fullPage: true });
	await page.goto('/');
	await page.locator('textarea').first().fill('fixture-error');
	await page.getByRole('button', { name: 'Send', exact: true }).click();
	await expect(page.getByText('Synthetic busy error', { exact: false }).first()).toBeVisible();
	await page.setViewportSize({ width: 390, height: 844 });
	await page.goto('/');
	await expect(page.locator('textarea').first()).toBeVisible();
	expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
	await page.screenshot({ path: test.info().outputPath('mobile.png'), fullPage: true });
	expect(errors).toEqual([]);
});

test('settings direct load, navigation, saved sampling and explicit unsupported-control error', async ({
	page
}) => {
	const errors: string[] = [];
	page.on('pageerror', (e) => errors.push(e.message));
	await page.goto('/#/settings');
	await expect(page).toHaveURL(/#\/settings\/general$/);
	await expect(
		page.getByText('Use LLM to generate conversation title', { exact: true })
	).toBeVisible();
	await page.screenshot({ path: test.info().outputPath('settings.png'), fullPage: true });
	await page.getByRole('link', { name: 'Sampling', exact: true }).click();
	await page.locator('#temperature').fill('0.5');
	await page.locator('#temperature').blur();
	await page.getByRole('button', { name: /Save/i }).click();
	await page.goto('/#/settings/sampling');
	await expect(page.locator('#temperature')).toHaveValue('0.5');
	await page.goto('/');
	await page.locator('textarea').first().fill('unsupported sampler');
	await page.getByRole('button', { name: 'Send', exact: true }).click();
	await expect(
		page.getByText('temperature: only greedy decoding (0) is supported', { exact: false }).first()
	).toBeVisible();
	await page.goto('/#/settings/sampling');
	await page.locator('#temperature').fill('0');
	await page.locator('#temperature').blur();
	await page.getByRole('button', { name: /Save/i }).click();
	await page.goto('/#/settings/general');
	await page.getByRole('link', { name: 'Import/Export', exact: true }).click();
	await expect(page.getByRole('button', { name: /Export/i }).first()).toBeVisible();
	expect(errors).toEqual([]);
});

import { test, expect } from '@playwright/test';

test('Go System One playground submits a bounded decision request', async ({ page }) => {
	const errors: string[] = [];
	page.on('pageerror', (e) => errors.push(e.message));
	await page.emulateMedia({ colorScheme: 'light' });
	await page.goto('/go-system-one');
	await expect(page.getByRole('heading', { name: 'Decision playground', exact: true })).toBeVisible();
	await expect(page.locator('#status')).toContainText('ready');
	await page.locator('#api').selectOption('decision');
	await page.getByRole('button', { name: 'Run decision' }).click();
	await expect(page.locator('.decision').first()).toContainText('"urgent": true');
	const firstContext = page.locator('.card').first();
	await expect(firstContext.locator('.field')).toHaveCount(2);
	await expect(firstContext.locator('.candidate')).toHaveCount(5);
	await expect(firstContext.locator('.candidate[data-field="urgent"]')).toHaveCount(2);
	await expect(firstContext.locator('.candidate[data-field="severity"]')).toHaveCount(3);
	await expect(firstContext.locator('.candidate[data-field="urgent"][data-value="true"]')).toContainText('87.50%');
	await expect(firstContext.locator('.candidate[data-field="urgent"][data-value="false"]')).toContainText('12.50%');
	await expect(firstContext.locator('.candidate[data-field="severity"][data-value="\\"low\\""]')).toContainText('10.00%');
	await expect(firstContext.locator('.candidate[data-field="severity"][data-value="\\"medium\\""]')).toContainText('20.00%');
	await expect(firstContext.locator('.candidate[data-field="severity"][data-value="\\"high\\""]')).toContainText('70.00%');
	await expect(firstContext.locator('.candidate[data-selected="true"]')).toHaveCount(2);
	await expect(page.locator('.card')).toHaveCount(2);
	await expect(page.locator('.pill').first()).toContainText('total: 3.50 ms');
	const style = await page.evaluate(() => {
		const root = getComputedStyle(document.documentElement);
		const body = getComputedStyle(document.body);
		const panel = getComputedStyle(document.querySelector('.panel')!);
		const textarea = getComputedStyle(document.querySelector('textarea')!);
		return {
			radius: root.getPropertyValue('--radius').trim(),
			bodyFont: body.fontFamily,
			panelRadius: panel.borderRadius,
			textareaFont: textarea.fontFamily,
			pageWidth: document.documentElement.scrollWidth,
			viewportWidth: innerWidth,
			colorScheme: root.colorScheme,
			background: body.backgroundColor
		};
	});
	expect(style.radius).toBe('.625rem');
	expect(style.bodyFont).toContain('ui-sans-serif');
	expect(style.panelRadius).toBe('14px');
	expect(style.textareaFont).toContain('ui-monospace');
	expect(style.pageWidth).toBeLessThanOrEqual(style.viewportWidth);
	expect(style.colorScheme).toBe('light');
	await page.screenshot({ path: test.info().outputPath('go-system-one-light.png'), fullPage: true });
	await page.setViewportSize({ width: 390, height: 844 });
	await page.emulateMedia({ colorScheme: 'dark' });
	await page.goto('/go-system-one');
	await page.locator('#api').selectOption('decision');
	await page.getByRole('button', { name: 'Run decision' }).click();
	await expect(page.locator('.decision').first()).toContainText('"urgent": true');
	await expect(page.locator('.card').first().locator('.candidate')).toHaveCount(5);
	await expect(page.locator('.card').first().locator('.candidate .prob')).toHaveText([
		'10.00%',
		'20.00%',
		'70.00%',
		'87.50%',
		'12.50%'
	]);
	expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
	expect(await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme)).toBe('dark');
	expect(await page.evaluate(() => getComputedStyle(document.body).backgroundColor)).not.toBe(style.background);
	expect(await page.locator('.grid').evaluate((el) => getComputedStyle(el).gridTemplateColumns.split(' ').length)).toBe(1);
	await page.screenshot({ path: test.info().outputPath('go-system-one-dark-mobile.png'), fullPage: true });
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

import { test, expect } from '@playwright/test';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';

async function assertNoOverflow(page: import('@playwright/test').Page) {
	expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
}

test('submits bounded decisions and follows the OS colour scheme', async ({ page }) => {
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(error.message));
	await page.emulateMedia({ colorScheme: 'light' });
	await page.goto('/go-system-one');
	await expect(page.getByRole('heading', { name: 'Decision playground', exact: true })).toBeVisible();
	await expect(page.locator('#status')).toContainText('ready');
	await page.getByLabel('API', { exact: true }).selectOption('decision');
	const submitted = page.waitForRequest(r => r.url().endsWith('/v1/decision'));
	await page.getByRole('button', { name: 'Run decision' }).click();
	expect((await submitted).postDataJSON()).toMatchObject({ mode: 'auto', tree_max: 128, cache_prompt: true });
	expect((await submitted).postDataJSON()).not.toHaveProperty('questions');
	await expect(page.locator('.decision')).toHaveCount(2);
	await expect(page.locator('.decision').first()).toContainText('"urgent": true');
	await expect(page.locator('.decision').nth(1)).toContainText('"urgent": false');
	const firstContext = page.locator('.card').first();
	await expect(firstContext.locator('.field')).toHaveCount(2);
	await expect(firstContext.locator('.candidate')).toHaveCount(5);
	await expect(firstContext.locator('.candidate[data-field="urgent"][data-value="true"]')).toContainText('87.50%');
	await expect(firstContext.locator('.candidate[data-field="urgent"][data-value="false"]')).toContainText('12.50%');
	await expect(firstContext.locator('.candidate[data-field="severity"][data-value="\\"low\\""]')).toContainText('10.00%');
	await expect(firstContext.locator('.candidate[data-field="severity"][data-value="\\"medium\\""]')).toContainText('20.00%');
	await expect(firstContext.locator('.candidate[data-field="severity"][data-value="\\"high\\""]')).toContainText('70.00%');
	await expect(firstContext.locator('.candidate[data-selected="true"]')).toHaveCount(2);
	await expect(page.locator('#timings')).toContainText('total: 3.50 ms');
	await assertNoOverflow(page);

	const light = await page.evaluate(() => ({
		colourScheme: getComputedStyle(document.documentElement).colorScheme,
		font: getComputedStyle(document.body).fontFamily,
		textareaFont: getComputedStyle(document.querySelector('textarea')!).fontFamily,
		panelRadius: getComputedStyle(document.querySelector('.panel')!).borderRadius,
		background: getComputedStyle(document.body).backgroundColor
	}));
	expect(light.colourScheme).toBe('light');
	expect(light.font).toContain('ui-sans-serif');
	expect(light.textareaFont).toContain('ui-monospace');
	expect(light.panelRadius).toBe('14px');

	await page.setViewportSize({ width: 390, height: 844 });
	await page.emulateMedia({ colorScheme: 'dark' });
	await page.goto('/go-system-one');
	await page.getByLabel('API', { exact: true }).selectOption('decision');
	await page.getByRole('button', { name: 'Run decision' }).click();
	await expect(page.locator('.decision')).toHaveCount(2);
	await expect(page.locator('.card').first().locator('.candidate .prob')).toHaveText([
		'10.00%',
		'20.00%',
		'70.00%',
		'87.50%',
		'12.50%'
	]);
	await assertNoOverflow(page);
	expect(await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme)).toBe('dark');
	expect(await page.evaluate(() => getComputedStyle(document.body).backgroundColor)).not.toBe(light.background);
	expect(await page.locator('.grid').evaluate((element) => getComputedStyle(element).gridTemplateColumns.split(' ').length)).toBe(1);
	expect(errors).toEqual([]);
});

test('keeps status and method boundaries intact', async ({ request }) => {
	const status = await request.get('/go-system-one/v1/status');
	expect(status.ok()).toBe(true);
	expect(await status.json()).toMatchObject({
		model: 'fixture-model',
		backend: 'fixture',
		device: 'synthetic',
		busy: false
	});
	const method = await request.post('/go-system-one/v1/status');
	expect(method.status()).toBe(405);
	expect(method.headers().allow).toBe('GET, HEAD');
});

for (const variant of [
	{ name: 'light-desktop', colour: 'light' as const, viewport: { width: 1280, height: 900 } },
	{ name: 'dark-mobile', colour: 'dark' as const, viewport: { width: 390, height: 844 } }
]) {
	test(`renders TypeSafe answers and usage in ${variant.name}`, async ({ page }) => {
		const errors: string[] = [];
		page.on('pageerror', e => errors.push(e.message));
		await page.setViewportSize(variant.viewport);
		await page.emulateMedia({ colorScheme: variant.colour });
		await page.goto('/go-system-one');
		await expect(page.locator('#status')).toContainText('ready');
		await expect(page.getByLabel('API', { exact: true })).toHaveValue('systemone');
		await expect(page.locator('#mode')).toBeHidden();
		await expect(page.locator('#treeMax')).toBeHidden();
		const submitted = page.waitForRequest(r => r.url().endsWith('/v1/systemone'));
		await page.getByRole('button', { name: 'Run questions' }).click();
		const body = (await submitted).postDataJSON();
		expect(Object.keys(body).sort()).toEqual(['model', 'questions', 'state']);
		expect(body.state).toEqual({ ticket: 'Production services are down and customers cannot connect.' });
		expect(Object.keys(body.questions)).toEqual(['urgent', 'team', 'severity']);
		await expect(page.locator('.field')).toHaveCount(3);
		expect(await page.locator('.field').evaluateAll(els => els.map(el => el.getAttribute('data-question')))).toEqual(['urgent', 'team', 'severity']);
		const noul = page.locator('[data-type="noul"]');
		await expect(noul.locator('.answer-value')).toHaveText('87.50% probability of yes');
		await expect(noul.locator('.candidate .prob')).toHaveText(['87.50%', '12.50%']);
		await expect(noul).not.toContainText('Confidence');
		await expect(noul.locator('[data-selected="true"]')).toHaveCount(0);
		const choice = page.locator('[data-type="choice"]');
		await expect(choice.locator('.answer-value')).toHaveText('operations');
		await expect(choice.locator('.candidate-value')).toHaveText(['operations', 'billing']);
		await expect(choice.locator('.candidate .prob')).toHaveText(['85.00%', '15.00%']);
		await expect(choice.locator('.candidate-note')).toHaveText('Confidence (local): 70.00%');
		await expect(choice.locator('[data-selected="true"]')).toHaveCount(1);
		const score = page.locator('[data-type="score"]');
		await expect(score.locator('.answer-value')).toHaveText('1.30 expected level');
		await expect(score.locator('.candidate-value')).toHaveText(['0 · Routine request', '1 · Degraded service', '2 · Total outage']);
		await expect(score.locator('.candidate .prob')).toHaveText(['20.00%', '30.00%', '50.00%']);
		await expect(score.locator('[data-selected="true"]')).toHaveCount(0);
		await expect(score.locator('.candidate-note')).toContainText('Confidence (local): 65.00%');
		await expect(page.locator('#timings')).toContainText('client HTTP:');
		await expect(page.locator('#timings')).toContainText('input tokens: 176');
		await expect(page.locator('#timings')).toContainText('output tokens: 120');
		await expect(page.locator('#timings')).not.toContainText('scoring:');
		await expect(page.locator('#run')).toBeEnabled();
		await assertNoOverflow(page);
		expect(await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme)).toBe(variant.colour);
		expect(errors).toEqual([]);
		if (process.env.UPDATE_PLAYGROUND_SCREENSHOTS === '1') {
			// Capture the real renderer without presenting random fixture HTTP time
			// as a benchmark. The server's status explicitly identifies the fixture.
			await page.locator('#timings .pill').first().evaluate(el => { el.textContent = 'client HTTP: fixture timing'; });
			const path = resolve('..', '..', 'docs', 'images');
			await mkdir(path, { recursive: true });
			await page.mouse.move(0, 0);
			await page.evaluate(() => { if (document.activeElement instanceof HTMLElement) document.activeElement.blur(); });
			await page.screenshot({ path: resolve(path, `go-system-one-${variant.name}.png`), fullPage: true, animations: 'disabled', caret: 'hide', style: '* { transition: none !important; }' });
		}
	});
}

test('preserves route drafts, submits text and null state, and clears stale results', async ({ page }) => {
	await page.goto('/go-system-one');
	await page.locator('#stateFormat').selectOption('text');
	await page.locator('#state').fill('A plain text ticket.');
	let submitted = page.waitForRequest(r => r.url().endsWith('/v1/systemone'));
	await page.getByRole('button', { name: 'Run questions' }).click();
	expect((await submitted).postDataJSON().state).toBe('A plain text ticket.');
	await expect(page.locator('.field')).toHaveCount(3);
	await page.locator('#api').selectOption('decision');
	await expect(page.locator('.field')).toHaveCount(0);
	await page.locator('#contexts').fill('Saved batch draft');
	await page.locator('#api').selectOption('systemone');
	await expect(page.locator('#state')).toHaveValue('A plain text ticket.');
	await expect(page.locator('#stateFormat')).toHaveValue('text');
	await page.locator('#stateFormat').selectOption('json');
	await page.locator('#state').fill('null');
	submitted = page.waitForRequest(r => r.url().endsWith('/v1/systemone'));
	await page.getByRole('button', { name: 'Run questions' }).click();
	expect((await submitted).postDataJSON().state).toBeNull();
	await expect(page.locator('.field')).toHaveCount(3);
	await page.locator('#state').fill('{');
	let calls = 0;
	page.on('request', r => { if (r.url().endsWith('/v1/systemone')) calls++; });
	await page.getByRole('button', { name: 'Run questions' }).click();
	await expect(page.getByRole('alert')).toContainText('State JSON:');
	await expect(page.locator('.field')).toHaveCount(0);
	expect(calls).toBe(0);
	await page.getByRole('button', { name: 'Reset', exact: true }).click();
	await expect(page.locator('#state')).toContainText('Production services');
	await expect(page.locator('#error')).toBeEmpty();
	await page.locator('#questions').fill('[]');
	await page.getByRole('button', { name: 'Run questions' }).click();
	await expect(page.getByRole('alert')).toHaveText('Questions must be a nonempty object.');
	expect(calls).toBe(0);
	await page.locator('#api').selectOption('decision');
	await expect(page.locator('#contexts')).toHaveValue('Saved batch draft');
});

test('locks the request while scoring and recovers from busy errors', async ({ page }) => {
	let release!: () => void;
	const wait = new Promise<void>(resolve => { release = resolve; });
	await page.route('**/v1/systemone', async route => {
		await wait;
		await route.fulfill({ status: 429, contentType: 'text/plain', body: 'decision inference busy; retry later' });
	});
	await page.goto('/go-system-one');
	await page.getByRole('button', { name: 'Run questions' }).click();
	await expect(page.locator('#api')).toBeDisabled();
	await expect(page.locator('#reset')).toBeDisabled();
	await expect(page.locator('#questions')).toBeDisabled();
	await expect(page.locator('#results')).toHaveAttribute('aria-busy', 'true');
	release();
	await expect(page.getByRole('alert')).toContainText('busy; retry later');
	await expect(page.locator('#api')).toBeEnabled();
	await expect(page.locator('#reset')).toBeEnabled();
	await expect(page.locator('#results')).toHaveAttribute('aria-busy', 'false');
	await page.unroute('**/v1/systemone');
	await page.getByRole('button', { name: 'Run questions' }).click();
	await expect(page.locator('.field')).toHaveCount(3);
	await expect(page.locator('#error')).toBeEmpty();
});

test('renders hostile labels and structured score legends as text', async ({ page }) => {
	const hostile = '<img src=x onerror="window.injected=true">';
	await page.route('**/v1/systemone', route => route.fulfill({ json: {
		model: hostile,
		answers: {
			[hostile]: { type: 'choice', choice: hostile, confidence: .5, probabilities: { [hostile]: .75, other: .25 } },
			rating: { type: 'score', score: .5, confidence: .5, probabilities: { '10': .25, '2': .25, '0': .5 }, legend: { '10': { rule: hostile }, '2': [hostile], '0': null } }
		}, usage: { input_tokens: 20, output_tokens: 10 }
	} }));
	await page.goto('/go-system-one');
	await page.locator('#questions').fill(JSON.stringify({ [hostile]: { type: 'choice', criteria: { [hostile]: null, other: null } }, rating: { type: 'score', criteria: [null, 'moderate'] } }));
	await page.getByRole('button', { name: 'Run questions' }).click();
	await expect(page.locator('.field')).toHaveCount(2);
	await expect(page.locator('#results img')).toHaveCount(0);
	await expect(page.locator('[data-type="choice"] .answer-value')).toHaveText(hostile);
	await expect(page.locator('[data-type="score"] .candidate-value')).toHaveText(['0', `2 · ["${hostile.replaceAll('"', '\\"')}"]`, `10 · {"rule":"${hostile.replaceAll('"', '\\"')}"}`]);
	expect(await page.evaluate(() => (window as any).injected)).toBeUndefined();
	await page.setViewportSize({ width: 390, height: 844 });
	await assertNoOverflow(page);
});

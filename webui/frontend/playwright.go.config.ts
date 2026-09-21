import { defineConfig } from '@playwright/test';

// Exercises the embedded assets and compatibility adapter, without inference.
export default defineConfig({
	testDir: './tests/go',
	fullyParallel: false,
	workers: 1,
	use: {
		baseURL: 'http://127.0.0.1:18181',
		headless: true,
		launchOptions: { args: ['--no-sandbox'] }
	},
	webServer: {
		command:
			'cd ../.. && WEBUI_BROWSER_TEST=1 GO_PHERENCE_DISABLE_NVIDIA=1 go test ./webui -run ^TestBrowserServer$ -count=1 -timeout=5m',
		url: 'http://127.0.0.1:18181',
		timeout: 120000,
		reuseExistingServer: false
	}
});

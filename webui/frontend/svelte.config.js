import { mdsvex } from 'mdsvex';
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

// Go embeds this output. Keep the upstream override for developer builds.
const outDir = process.env.LLAMA_UI_OUT_DIR ?? '../dist';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: [vitePreprocess(), mdsvex()],

	kit: {
		// No wall-clock build ID: checked-in assets must be reproducible.
		version: { name: 'llama-4a6735f1-go-1' },
		paths: {
			relative: true
		},
		router: { type: 'hash' },
		adapter: adapter({
			pages: outDir,
			assets: outDir,
			fallback: 'index.html',
			precompress: false,
			strict: true
		}),
		output: {
			bundleStrategy: 'single'
		},
		alias: {
			$styles: 'src/styles'
		},
		version: {
			name: 'llama-ui'
		}
	},

	extensions: ['.svelte', '.svx']
};

export default config;

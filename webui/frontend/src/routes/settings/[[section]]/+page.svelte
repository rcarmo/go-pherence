<script lang="ts">
	import { SettingsChat } from '$lib/components/app/settings';
	import { page } from '$app/state';
	import { replaceState } from '$app/navigation';
	import { RouterService } from '$lib/services';
	import { SETTINGS_SECTION_SLUGS } from '$lib/constants';
	import { onMount, tick } from 'svelte';

	onMount(() => {
		if (!page.params.section) {
			// Direct hash loads mount before SvelteKit assigns its root instance.
			void tick().then(() => {
				replaceState(RouterService.settings(SETTINGS_SECTION_SLUGS.GENERAL), {});
			});
		}
	});
</script>

<SettingsChat initialSection={(page.params as Record<string, string | undefined>).section} />

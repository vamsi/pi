/**
 * Dark/light theme toggle.
 */

const STORAGE_KEY = 'pi-web-theme';

/** Initialize theme from stored preference or system default */
export function initTheme() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'light') {
        document.documentElement.classList.remove('dark');
    } else if (stored === 'dark') {
        document.documentElement.classList.add('dark');
    } else {
        // System preference
        if (window.matchMedia('(prefers-color-scheme: light)').matches) {
            document.documentElement.classList.remove('dark');
        } else {
            document.documentElement.classList.add('dark');
        }
    }
}

/** Toggle between dark and light */
export function toggleTheme() {
    const isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem(STORAGE_KEY, isDark ? 'dark' : 'light');
    return isDark;
}

"""Playwright validation of the obs-table click behavior."""
import asyncio
from playwright.async_api import async_playwright

SPACE = "https://deepearth-california-flourishing-pollination.hf.space/"

async def main() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1600, "height": 1100})
        page = await ctx.new_page()
        # Capture console + network for debugging
        page.on("console", lambda m: print(f"[console.{m.type}] {m.text}"))
        page.on("requestfailed", lambda r: print(f"[reqfail] {r.url} {r.failure}"))
        # Look for /api/predict or /gradio_api/queue/* calls
        api_calls = []
        page.on("request", lambda r: api_calls.append(r.url) if ("queue" in r.url or "predict" in r.url) else None)

        print(f"navigating to {SPACE}", flush=True)
        await page.goto(SPACE, wait_until="domcontentloaded", timeout=60000)
        # No iframe — we go to the Space subdomain directly.
        await page.wait_for_timeout(12000)  # let gradio mount
        frame = page  # use the page itself as the "frame" handle

        await page.screenshot(path="/tmp/space_loaded.png", full_page=True)
        print("screenshot: /tmp/space_loaded.png", flush=True)

        # Click "Random across all species" to populate obs_table
        print("clicking 'Random across all species'…", flush=True)
        try:
            random_btn = frame.locator("button:has-text('Random across all species')")
            await random_btn.click(timeout=10000)
            print("  clicked", flush=True)
        except Exception as e:
            print(f"  random click FAILED: {e}", flush=True)

        # Wait for the table to populate
        await page.wait_for_timeout(10000)
        await page.screenshot(path="/tmp/space_post_random.png", full_page=True)
        print("screenshot: /tmp/space_post_random.png", flush=True)

        # Dump anything that looks like a list/table
        print("\n=== scanning DOM for dataset-like containers ===", flush=True)
        dom_dump = await frame.evaluate("""() => {
            const out = [];
            for (const sel of ['table', '.dataset-content', '.examples-holder',
                                 '.svelte-component', '[class*="dataset"]',
                                 '[class*="example"]', '[class*="gallery"]']) {
                document.querySelectorAll(sel).forEach(el => {
                    if (el.children.length > 0) {
                        out.push({sel, className: el.className.slice(0, 100),
                                  childCount: el.children.length,
                                  first: el.children[0] ? (el.children[0].tagName + '.' + el.children[0].className.slice(0,80)) : ''});
                    }
                });
            }
            return out.slice(0, 15);
        }""")
        for d in dom_dump: print(f"  {d}", flush=True)

        # Dump all elements that look like rows
        print("\n=== row-like elements ===", flush=True)
        rows = await frame.evaluate("""() => {
            const out = [];
            // gr.Dataset typically uses .gallery-item, .example, or table rows
            for (const sel of ['button', 'tr', '[role="button"]', '.example']) {
                document.querySelectorAll(sel).forEach((el, i) => {
                    if (i < 30) {
                        const text = (el.innerText || '').slice(0, 60);
                        out.push({sel, text, cls: el.className.slice(0, 60)});
                    }
                });
            }
            return out;
        }""")
        for r in rows[:40]: print(f"  {r}", flush=True)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())

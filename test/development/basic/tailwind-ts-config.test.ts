import { join } from 'path'
import webdriver, { Playwright } from 'next-webdriver'
import { FileRef, nextTestSetup } from 'e2e-utils'

describe('TailwindCSS with TypeScript config imports', () => {
  const { next } = nextTestSetup({
    files: {
      'postcss.config.js': new FileRef(
        join(__dirname, 'tailwind-ts-config/postcss.config.js')
      ),
      'tailwind.config.ts': new FileRef(
        join(__dirname, 'tailwind-ts-config/tailwind.config.ts')
      ),
      lib: new FileRef(join(__dirname, 'tailwind-ts-config/lib')),
      pages: new FileRef(join(__dirname, 'tailwind-ts-config/pages')),
      styles: new FileRef(join(__dirname, 'tailwind-ts-config/styles')),
    },
    dependencies: {
      tailwindcss: '^3.4.1',
      postcss: '^8',
      autoprefixer: '^10',
    },
  })

  // This test verifies that importing TypeScript files from tailwind.config.ts
  // doesn't produce "Module not found" warnings in Turbopack dev mode.
  // Regression test for https://github.com/vercel/next.js/issues/87898
  it('should not produce module resolution warnings for TS imports in tailwind config', async () => {
    let browser: Playwright
    try {
      browser = await webdriver(next.url, '/')

      // Verify the page loads and tailwind styles are applied
      const heading = await browser.elementByCss('#heading')
      expect(await heading.text()).toBe('Hello World')

      // Verify tailwind CSS is working (text-blue-600)
      const color = await heading.getComputedCss('color')
      expect(color).toBe('rgb(37, 99, 235)')

      // Check CLI output for module resolution warnings
      // The bug caused warnings like: "Module not found: Can't resolve './lib/plugin'"
      const cliOutput = next.cliOutput
      expect(cliOutput).not.toContain("Can't resolve './lib/plugin'")
      expect(cliOutput).not.toContain('Module not found')
    } finally {
      if (browser) {
        await browser.close()
      }
    }
  })
})

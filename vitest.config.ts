import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: false,
    environment: "node",
    reporters: process.env.CI ? ["default", "junit"] : ["default"],
    outputFile: process.env.CI ? "vitest-report.xml" : undefined,
    coverage: {
      provider: "v8",
      reporter: process.env.CI ? ["text", "lcov", "json"] : ["text"],
      reportsDirectory: "coverage",
      include: ["src/**/*.ts"],
      exclude: [
        "src/tests/**",
        "src/**/*.test.ts",
        "src/**/*.d.ts",
        "src/**/index.ts",
      ],
      thresholds: {
        branches: 40,
        functions: 45,
        lines: 40,
        statements: 40,
      },
    },
    testTimeout: 15_000,
    hookTimeout: 10_000,
  },
});

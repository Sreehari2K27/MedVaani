// Minimal ambient declaration for the single Node built-in used by the
// dependency-free test script (src/mi/miChat.test.ts). Keeps `npm run build`
// (tsc) green without adding @types/node as a dependency.
declare module "node:fs" {
  export function readFileSync(
    path: string | URL,
    encoding: "utf-8",
  ): string;
}

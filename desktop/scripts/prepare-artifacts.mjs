import { chmodSync, cpSync, existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const desktopDir = path.resolve(__dirname, "..");
const repoRoot = path.resolve(desktopDir, "..");
const frontendDir = path.join(repoRoot, "frontend");
const backendDir = path.join(repoRoot, "backend");
const artifactsDir = path.join(desktopDir, ".artifacts");
const rendererArtifactsDir = path.join(artifactsDir, "renderer");
const backendArtifactsDir = path.join(artifactsDir, "backend");
const runtimeArtifactsDir = path.join(artifactsDir, "runtime");

function run(command, args, options = {}) {
  const resolvedCommand =
    process.platform === "win32" && command === "npm" ? "npm.cmd" : command;
  const result = spawnSync(resolvedCommand, args, {
    stdio: "inherit",
    cwd: options.cwd || repoRoot,
    env: options.env || process.env,
  });
  if (result.status !== 0) {
    throw new Error(`${resolvedCommand} ${args.join(" ")} failed with exit code ${result.status}`);
  }
}

function capture(command, args, options = {}) {
  const resolvedCommand =
    process.platform === "win32" && command === "npm" ? "npm.cmd" : command;
  const result = spawnSync(resolvedCommand, args, {
    stdio: ["ignore", "pipe", "pipe"],
    cwd: options.cwd || repoRoot,
    env: options.env || process.env,
    encoding: "utf-8",
  });
  if (result.status !== 0) {
    const detail = (result.stderr || result.stdout || "").trim();
    throw new Error(`${resolvedCommand} ${args.join(" ")} failed: ${detail}`);
  }
  return String(result.stdout || "").trim();
}

function resolveBackendPython() {
  const override = process.env.AGENT_PLAYGROUND_PYTHON_BIN;
  if (override && existsSync(override)) return override;

  const candidates = [
    path.join(backendDir, ".venv", "bin", "python"),
    path.join(backendDir, ".venv", "Scripts", "python.exe"),
  ];
  return candidates.find((candidate) => existsSync(candidate)) || null;
}

function removeAndRecreate(dirPath) {
  rmSync(dirPath, { recursive: true, force: true });
  mkdirSync(dirPath, { recursive: true });
}

function copyFrontendArtifacts() {
  const frontendDistDir = path.join(frontendDir, "dist");
  if (existsSync(frontendDistDir)) {
    console.log("SKIP: frontend/dist already exists, skipping npm run build");
  } else {
    run("npm", ["run", "build"], {
      cwd: frontendDir,
      env: { ...process.env, AGENT_PLAYGROUND_DESKTOP_BUILD: "1" },
    });
  }
  if (!existsSync(frontendDistDir)) {
    throw new Error("Frontend build output not found: frontend/dist");
  }
  cpSync(frontendDistDir, rendererArtifactsDir, { recursive: true });
}

function resolveBundledNodeRuntime() {
  const nodeBinary = process.execPath;
  if (!existsSync(nodeBinary)) {
    throw new Error(`Node executable not found: ${nodeBinary}`);
  }

  // Prefer NPM_GLOBAL_ROOT from env (set by CI workflow via `npm root -g`),
  // falling back to calling `npm root -g` directly.
  let npmGlobalRoot = process.env.NPM_GLOBAL_ROOT || "";
  if (!npmGlobalRoot) {
    npmGlobalRoot = capture("npm", ["root", "-g"], { cwd: repoRoot });
  }
  const npmPackageDir = path.join(npmGlobalRoot, "npm");
  const npmCliPath = path.join(npmPackageDir, "bin", "npm-cli.js");
  if (!existsSync(npmPackageDir) || !existsSync(npmCliPath)) {
    throw new Error(`Bundled npm package not found under: ${npmPackageDir}`);
  }

  return {
    nodeBinary,
    npmPackageDir,
    npmCliPath,
  };
}

function parseSharedLibraryDeps(binaryPath) {
  if (process.platform === "darwin") {
    const output = capture("otool", ["-L", binaryPath], { cwd: repoRoot });
    return output
      .split("\n")
      .slice(1)
      .map((line) => line.trim().split(" ")[0])
      .filter(Boolean);
  }

  if (process.platform === "linux") {
    const output = capture("ldd", [binaryPath], { cwd: repoRoot });
    return output
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const match = line.match(/=>\s+(\S+)/);
        if (match) return match[1];
        const direct = line.match(/^(\S+)\s+\(/);
        return direct ? direct[1] : "";
      })
      .filter(Boolean);
  }

  return [];
}

function resolveDynamicDependency(dep, ownerPath) {
  if (!dep) return null;
  if (dep.startsWith("/usr/lib/") || dep.startsWith("/System/")) {
    return null;
  }
  if (path.isAbsolute(dep)) {
    return existsSync(dep) ? dep : null;
  }

  const ownerDir = path.dirname(ownerPath);
  const searchRoots = [
    ownerDir,
    path.resolve(ownerDir, "..", "lib"),
    path.resolve(ownerDir, "..", "lib", "node_modules"),
  ];
  const prefixes = ["@loader_path/", "@rpath/"];
  for (const prefix of prefixes) {
    if (!dep.startsWith(prefix)) continue;
    const suffix = dep.slice(prefix.length);
    for (const root of searchRoots) {
      const candidate = path.resolve(root, suffix);
      if (existsSync(candidate)) {
        return candidate;
      }
    }
  }
  return null;
}

function collectSharedLibraryDeps(entryPath) {
  if (process.platform === "win32") {
    return [];
  }

  const queue = [entryPath];
  const visited = new Set();
  const collected = [];
  while (queue.length) {
    const current = queue.pop();
    if (!current || visited.has(current)) {
      continue;
    }
    visited.add(current);
    const deps = parseSharedLibraryDeps(current);
    for (const dep of deps) {
      const resolved = resolveDynamicDependency(dep, current);
      if (!resolved || visited.has(resolved)) {
        continue;
      }
      collected.push(resolved);
      queue.push(resolved);
    }
  }
  return collected;
}

function writeExecutableScript(targetPath, content) {
  mkdirSync(path.dirname(targetPath), { recursive: true });
  writeFileSync(targetPath, content, "utf-8");
  if (process.platform !== "win32") {
    chmodSync(targetPath, 0o755);
  }
}

function copyBundledRuntimeArtifacts() {
  const { nodeBinary, npmPackageDir } = resolveBundledNodeRuntime();
  const runtimeNodeRoot = path.join(runtimeArtifactsDir, "node");
  const runtimeBinDir = path.join(runtimeNodeRoot, "bin");
  const runtimeLibDir = path.join(runtimeNodeRoot, "lib");
  mkdirSync(runtimeBinDir, { recursive: true });
  mkdirSync(runtimeLibDir, { recursive: true });

  const nodeTargetName = process.platform === "win32" ? "node.exe" : "node";
  const nodeTargetPath = path.join(runtimeBinDir, nodeTargetName);
  cpSync(nodeBinary, nodeTargetPath, { force: true });

  for (const depPath of collectSharedLibraryDeps(nodeBinary)) {
    cpSync(depPath, path.join(runtimeLibDir, path.basename(depPath)), { force: true });
  }

  const npmTargetDir = path.join(runtimeLibDir, "node_modules", "npm");
  cpSync(npmPackageDir, npmTargetDir, { recursive: true, force: true });

  if (process.platform === "win32") {
    const npmCmdPath = path.join(runtimeBinDir, "npm.cmd");
    const npxCmdPath = path.join(runtimeBinDir, "npx.cmd");
    writeExecutableScript(
      npmCmdPath,
      `@echo off\r\n"%~dp0node.exe" "%~dp0..\\lib\\node_modules\\npm\\bin\\npm-cli.js" %*\r\n`,
    );
    writeExecutableScript(
      npxCmdPath,
      `@echo off\r\n"%~dp0node.exe" "%~dp0..\\lib\\node_modules\\npm\\bin\\npx-cli.js" %*\r\n`,
    );
  } else {
    const npmScript = `#!/bin/sh
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$SCRIPT_DIR/node" "$SCRIPT_DIR/../lib/node_modules/npm/bin/npm-cli.js" "$@"
`;
    const npxScript = `#!/bin/sh
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$SCRIPT_DIR/node" "$SCRIPT_DIR/../lib/node_modules/npm/bin/npx-cli.js" "$@"
    `;
    const npmPath = path.join(runtimeBinDir, "npm");
    const npxPath = path.join(runtimeBinDir, "npx");
    writeExecutableScript(npmPath, npmScript);
    writeExecutableScript(npxPath, npxScript);
  }

  if (existsSync(nodeTargetPath) && process.platform !== "win32") {
    chmodSync(nodeTargetPath, 0o755);
  }
}

function buildBackendArtifacts() {
  const pythonBin = resolveBackendPython();
  if (!pythonBin) {
    throw new Error(
      "Backend virtualenv python not found. Create backend/.venv first, or set AGENT_PLAYGROUND_PYTHON_BIN.",
    );
  }

  const addDataSeparator = process.platform === "win32" ? ";" : ":";
  const specDir = path.join(artifactsDir, "pyinstaller-spec");
  const buildDir = path.join(artifactsDir, "pyinstaller-build");
  const pyinstallerConfigDir = path.join(artifactsDir, "pyinstaller-config");
  mkdirSync(specDir, { recursive: true });
  mkdirSync(buildDir, { recursive: true });
  mkdirSync(pyinstallerConfigDir, { recursive: true });

  run(
    pythonBin,
    [
      "-m",
      "PyInstaller",
      path.join(backendDir, "desktop_entry.py"),
      "--name",
      "agent-playground-backend",
      "--noconfirm",
      "--clean",
      "--onedir",
      "--paths",
      backendDir,
      "--distpath",
      backendArtifactsDir,
      "--workpath",
      buildDir,
      "--specpath",
      specDir,
      "--hidden-import",
      "uvicorn.logging",
      "--hidden-import",
      "uvicorn.loops.auto",
      "--hidden-import",
      "uvicorn.protocols.http.auto",
      "--hidden-import",
      "uvicorn.protocols.websockets.auto",
      "--hidden-import",
      "uvicorn.lifespan.on",
      "--add-data",
      `${path.join(backendDir, "skills")}${addDataSeparator}backend/skills`,
      "--add-data",
      `${path.join(backendDir, "data")}${addDataSeparator}backend/data`,
    ],
    {
      cwd: repoRoot,
      env: {
        ...process.env,
        PYINSTALLER_CONFIG_DIR: pyinstallerConfigDir,
      },
    },
  );
}

removeAndRecreate(artifactsDir);
copyFrontendArtifacts();
copyBundledRuntimeArtifacts();
buildBackendArtifacts();

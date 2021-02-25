/* eslint-disable no-restricted-syntax */
/* eslint-disable import/no-extraneous-dependencies */
/* eslint-disable node/no-unpublished-require */

const fs = require('fs');
const esbuild = require('esbuild');
const ts = require('typescript');
const log = require('@vladmandic/pilogger');

// keeps esbuild service instance cached
let es;
const banner = `
  /*
  Scaffold
  homepage: <https://github.com/vladmandic/scaffold>
  author: <https://github.com/vladmandic>'
  */
`;

// tsc configuration

const tsconfig = {
  noEmitOnError: false,
  target: ts.ScriptTarget.ES2018,
  module: ts.ModuleKind.ES2020,
  outDir: 'types/',
  declaration: true,
  emitDeclarationOnly: true,
  emitDecoratorMetadata: true,
  experimentalDecorators: true,
  skipLibCheck: true,
  strictNullChecks: true,
  baseUrl: './',
  paths: {
    tslib: ['node_modules/tslib/tslib.d.ts'],
  },
};

// common configuration
const common = {
  banner,
  minifyWhitespace: true,
  minifyIdentifiers: true,
  minifySyntax: true,
  bundle: true,
  sourcemap: true,
  logLevel: 'error',
  target: 'es2018',
  // tsconfig: './tsconfig.json',
};

const targets = {
  browserBundle: {
    esm: {
      platform: 'browser',
      format: 'esm',
      entryPoints: ['src/index.js'],
      outfile: 'dist/index.js',
      metafile: 'dist/index.json',
      external: ['fs', 'buffer', 'util', 'os', 'string_decoder'],
    },
  },
};

async function getStats(metafile) {
  const stats = {};
  if (!fs.existsSync(metafile)) return stats;
  const data = fs.readFileSync(metafile);
  const json = JSON.parse(data.toString());
  if (json && json.inputs && json.outputs) {
    for (const [key, val] of Object.entries(json.inputs)) {
      if (key.startsWith('node_modules')) {
        stats.modules = (stats.modules || 0) + 1;
        stats.moduleBytes = (stats.moduleBytes || 0) + val.bytes;
      } else {
        stats.imports = (stats.imports || 0) + 1;
        stats.importBytes = (stats.importBytes || 0) + val.bytes;
      }
    }
    const files = [];
    for (const [key, val] of Object.entries(json.outputs)) {
      if (!key.endsWith('.map')) {
        files.push(key);
        stats.outputBytes = (stats.outputBytes || 0) + val.bytes;
      }
    }
    stats.outputFiles = files.join(', ');
  }
  return stats;
}

function compile(fileNames, options) {
  log.info('Compile typings:', fileNames);
  const program = ts.createProgram(fileNames, options);
  const emit = program.emit();
  const diag = ts
    .getPreEmitDiagnostics(program)
    .concat(emit.diagnostics);
  for (const info of diag) {
    // @ts-ignore
    const msg = info.messageText.messageText || info.messageText;
    if (msg.includes('package.json')) continue;
    if (msg.includes('Expected 0 arguments, but got 1')) continue;
    if (info.file) {
      const pos = info.file.getLineAndCharacterOfPosition(info.start || 0);
      log.error(`TSC: ${info.file.fileName} [${pos.line + 1},${pos.character + 1}]:`, msg);
    } else {
      log.error('TSC:', msg);
    }
  }
}

// rebuild on file change
async function build(f, msg) {
  log.info('Build: file', msg, f, 'target:', common.target);
  if (!es) es = await esbuild.startService();
  // common build options
  try {
    // rebuild all target groups and types
    for (const [targetGroupName, targetGroup] of Object.entries(targets)) {
      for (const [targetName, targetOptions] of Object.entries(targetGroup)) {
        // if triggered from watch mode, rebuild only browser bundle
        if ((require.main !== module) && (targetGroupName !== 'browserBundle')) continue;
        await es.build({ ...common, ...targetOptions });
        const stats = await getStats(targetOptions.metafile);
        log.state(`Build for: ${targetGroupName} type: ${targetName}:`, stats);
      }
    }
  } catch (err) {
    // catch errors and print where it occured
    log.error('Build error', JSON.stringify(err.errors || err, null, 2));
    if (require.main === module) process.exit(1);
  }

  // generate typings
  // compile(targets.browserBundle.esm.entryPoints, tsconfig);

  if (require.main === module) process.exit(0);
}

if (require.main === module) {
  log.header();
  build('all', 'startup');
} else {
  exports.build = build;
}

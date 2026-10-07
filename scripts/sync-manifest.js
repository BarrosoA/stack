const fs = require('fs');
const path = require('path');

const docsDir = path.join(__dirname, '..', 'documents');
const manifestPath = path.join(docsDir, 'manifest.json');
const allowedExtensions = new Set(['.pdf', '.md', '.txt']);

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(0)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
}

function slugToTitle(filename) {
  const base = path.basename(filename, path.extname(filename));
  return base
    .replace(/[-_]+/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function syncManifest() {
  let existing = { documents: [] };
  if (fs.existsSync(manifestPath)) {
    try {
      existing = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    } catch (_) {
      existing = { documents: [] };
    }
  }

  const existingList = existing.documents || [];
  const existingByFile = new Map(
    existingList.map((doc, idx) => [doc.filename, { ...doc, _order: idx }])
  );

  const files = fs
    .readdirSync(docsDir)
    .filter((f) => allowedExtensions.has(path.extname(f).toLowerCase()))
    .sort((a, b) => {
      const prevA = existingByFile.get(a);
      const prevB = existingByFile.get(b);
      if (prevA && prevB) return prevA._order - prevB._order;
      if (prevA) return -1;
      if (prevB) return 1;
      const statA = fs.statSync(path.join(docsDir, a));
      const statB = fs.statSync(path.join(docsDir, b));
      return statA.mtimeMs - statB.mtimeMs;
    });

  const today = new Date().toISOString().slice(0, 10);

  const documents = files.map((filename) => {
    const fullPath = path.join(docsDir, filename);
    const stat = fs.statSync(fullPath);
    const ext = path.extname(filename).toLowerCase().slice(1);
    const prev = existingByFile.get(filename) || {};
    const id = path.basename(filename, path.extname(filename)).toLowerCase();

    return {
      id: prev.id || id,
      title: prev.title || slugToTitle(filename),
      subtitle: prev.subtitle || '',
      filename,
      path: `documents/${filename}`,
      type: ext,
      pages: prev.pages || null,
      size: formatBytes(stat.size),
      added: prev.added || stat.mtime.toISOString().slice(0, 10) || today,
      category: prev.category || 'Document'
    };
  });

  const output = {
    updatedAt: today,
    documents
  };

  fs.writeFileSync(manifestPath, JSON.stringify(output, null, 2) + '\n', 'utf8');
  console.log(`Synced ${documents.length} document(s) to documents/manifest.json`);
}

syncManifest();


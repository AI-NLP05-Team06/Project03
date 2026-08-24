const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const source = path.join(root, 'admin_ui_source');
const target = path.join(root, 'kdic_deploy_assets', 'kdic-admin-ui.html');
const index = fs.readFileSync(path.join(source, 'index.html'), 'utf8');
const css = fs.readFileSync(path.join(source, 'styles.css'), 'utf8');
const js = fs.readFileSync(path.join(source, 'admin.js'), 'utf8');
const bundled = index
  .replace('<link rel="stylesheet" href="styles.css">', `<style>\n${css}\n</style>`)
  .replace('<script src="admin.js"></script>', `<script>\n${js}\n</script>`);
if (bundled.includes('styles.css') || bundled.includes('src="admin.js"')) {
  throw new Error('Administrator UI bundling failed.');
}
fs.writeFileSync(target, bundled, 'utf8');
console.log(target);

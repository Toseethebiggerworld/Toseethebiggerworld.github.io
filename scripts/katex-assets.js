const fs = require('node:fs');
const path = require('node:path');

hexo.extend.generator.register('katex-assets', function () {
  const distDirectory = path.join(path.dirname(require.resolve('katex/package.json')), 'dist');
  const assets = ['katex.min.css'];

  for (const fontName of fs.readdirSync(path.join(distDirectory, 'fonts'))) {
    assets.push('fonts/' + fontName);
  }

  return assets.map(assetName => ({
    path: 'css/' + assetName,
    data: () => fs.createReadStream(path.join(distDirectory, assetName))
  }));
});

hexo.extend.generator.register('texmath-style', function () {
  return {
    path: 'css/texmath.css',
    data: () => fs.createReadStream(require.resolve('markdown-it-texmath/css/texmath.css'))
  };
});

const texmath = require('markdown-it-texmath');
const configuredParsers = new WeakSet();

hexo.extend.filter.register('markdown-it:renderer', function (parser) {
  if (configuredParsers.has(parser)) return;

  const inlineRule = texmath.rules.dollars.inline.find(rule => rule.name === 'math_inline');
  parser.inline.ruler.at('math_inline', texmath.inline({
    ...inlineRule,
    rex: /\$(?!\$)((?:\\[^\r\n]|[^$\\\r\n])+?)\$(?!\$)/gy
  }));

  for (const blockRule of texmath.rules.dollars.block) {
    parser.block.ruler.at(blockRule.name, texmath.block({
      ...blockRule,
      rex: new RegExp(blockRule.rex.source + '[ \\t]*$', blockRule.rex.flags)
    }));
  }

  configuredParsers.add(parser);
});

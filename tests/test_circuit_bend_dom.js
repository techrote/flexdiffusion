// Optional DOM integration test: provide linkedom through NODE_PATH.
let parseHTML;
try { ({parseHTML} = require('linkedom')); }
catch (error) {
  if (error.code !== 'MODULE_NOT_FOUND') throw error;
  console.log('Skipped optional circuit bend DOM test: linkedom is not installed');
  process.exit(0);
}
const fs = require('fs'), vm = require('vm'), assert = require('assert');
process.chdir(require('path').resolve(__dirname, '..'));
const fixture = fs.readFileSync('tests/fixtures/circuit-bend.html', 'utf8');
const {window} = parseHTML(fixture);
// linkedom has no select.value setter; emulate the browser form property.
Object.defineProperty(window.HTMLSelectElement.prototype, 'value', {
  get() { return (this.querySelector('option[selected]') || this.querySelector('option'))?.value || ''; },
  set(value) { for (const o of this.querySelectorAll('option')) { if(o.value === String(value)) o.setAttribute('selected', ''); else o.removeAttribute('selected'); } }
});
const scope = { window, document:window.document, Event:window.Event, CustomEvent:window.CustomEvent,
  MutationObserver:window.MutationObserver, console, seed:window.document.querySelector('#seed'),
  Option:function(text,value){const o=window.document.createElement('option');o.textContent=text;o.value=value;return o;} };
vm.createContext(scope);
vm.runInContext(fixture.match(/<script>([\s\S]*?)<\/script>/)[1], scope);
for(const name of ['circuit-bend', 'circuit-bend-grid']) vm.runInContext(fs.readFileSync(`ui/plugins/ui/${name}.plugin.js`,'utf8'), scope);
const field = id => scope.document.querySelector('#'+id);
const change = (id,value) => {field(id).value=value;field(id).dispatchEvent(new window.Event('change',{bubbles:true}));};
change('sampler_name','plms');
field('circuit_bend_enabled').checked=true;
change('circuit_bend_history_source','seed');
change('circuit_bend_blend_target','independent');
change('circuit_bend_blend_curve','oscillate');
change('circuit_bend_schedule','exact');
change('circuit_bend_donor_seed','5');
change('circuit_bend_blend_period','8');
change('num_inference_steps','16');
const task=scope.getCurrentUserRequest();
assert.equal(task.reqBody.circuit_bend.donor_seed,5);
assert.equal(task.reqBody.circuit_bend.blend_target,'independent');
assert.equal(task.reqBody.circuit_bend.blend_period,8);
assert.equal(field('circuit_bend_donor_fields').style.display,'grid');
assert(field('bend_grid_x').innerHTML.includes('blend_period'));
change('circuit_bend_donor_seed','22');
scope.restoreTaskToUI(task);
assert.equal(scope.getCurrentUserRequest().reqBody.circuit_bend.donor_seed,5);
assert.equal(scope.getCurrentUserRequest().reqBody.circuit_bend.blend_period,8);
change('sampler_name','ddim');
change('circuit_bend_eta','.8');
change('circuit_bend_memory','1');
change('circuit_bend_seed','123');
const ddim=scope.getCurrentUserRequest();
assert.equal(ddim.reqBody.circuit_bend.mode,'ddim');
assert.equal(ddim.reqBody.circuit_bend.eta,.8);
assert.equal(field('circuit_bend_noise_fields').style.display,'grid');
assert.equal(field('circuit_bend_history_fields').style.display,'none');
assert.equal(field('circuit_bend_churn').style.display,'none');
assert(field('bend_grid_x').innerHTML.includes('eta'));
change('circuit_bend_eta','.1');scope.restoreTaskToUI(ddim);
assert.equal(scope.getCurrentUserRequest().reqBody.circuit_bend.eta,.8);
change('backend','ed_diffusers');
assert.equal(scope.getCurrentUserRequest().reqBody.circuit_bend,undefined);
change('backend','ed_classic');
field('circuit_bend_enabled').checked=false;
assert.equal(scope.getCurrentUserRequest().reqBody.circuit_bend,undefined);
console.log('DOM integration: new controls, sampler gating, grid axes, request generation and restore passed');
change('sampler_name','plms');
field('circuit_bend_enabled').checked=true;
field('circuit_bend_mix_enabled').checked=true;
field('circuit_bend_mix_enabled').dispatchEvent(new window.Event('change',{bubbles:true}));
let hybrid=scope.getCurrentUserRequest();
assert.equal(hybrid.reqBody.circuit_bend.mode,'sampler_mix');
assert.equal(hybrid.reqBody.circuit_bend.eta,0);
assert.equal(hybrid.reqBody.circuit_bend.schedule,'exact');
assert.equal(field('circuit_bend_history_fields').style.display,'none');
assert.equal(field('circuit_bend_mix_fields').style.display,'');
assert(field('bend_grid_x').innerHTML.includes('mix_block'));
change('circuit_bend_mix_method','coupled');
change('circuit_bend_mix_exchange','.35');
change('circuit_bend_mix_curve','oscillate');
hybrid=scope.getCurrentUserRequest();
assert(field('bend_grid_x').innerHTML.includes('mix_exchange'));
assert.equal(field('circuit_bend_mix_first').disabled,true);
change('circuit_bend_mix_exchange','.8');
scope.restoreTaskToUI(hybrid);
assert.equal(scope.getCurrentUserRequest().reqBody.circuit_bend.mix_exchange,.35);
assert.equal(scope.getCurrentUserRequest().reqBody.circuit_bend.mix_curve,'oscillate');
assert.equal(field('circuit_bend_mix_enabled').checked,true);
scope.restoreTaskToUI(task);
assert.equal(field('circuit_bend_mix_enabled').checked,false);
assert.equal(scope.getCurrentUserRequest().reqBody.circuit_bend.mode,'plms_history');
console.log('DOM integration: sampler mixing toggle, alternation/coupling controls, grids and recipe restore passed');

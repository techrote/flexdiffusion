const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {timeline, gifHeader, gifFrame} = require('../scripts/step_viewer.js');
const order = timeline(20, 8);
assert.deepEqual(order.map(x => x.step), [...Array.from({length:20},(_,i)=>i+1), ...Array.from({length:19},(_,i)=>19-i)]);
assert.equal(order[19].delay,100); assert.equal(order[38].delay,100);
assert.equal(order[0].delay,13);
assert.equal(timeline(20,20)[0].delay,5);
assert.equal(timeline(20,8,true).length,38);
assert.equal(timeline(20,8,true)[0].delay,100);
assert.equal(timeline(20,8,true)[19].delay,100);
assert.equal(timeline(16,8,true).length,30);
assert.equal(timeline(16,8,false).length,31);
assert.equal(timeline(16,8,true)[15].step,16);
assert.equal(timeline(16,8,true)[15].delay,100);
assert.equal(Buffer.from(gifHeader(512,512,true)).includes(Buffer.from('NETSCAPE2.0')),true);
assert.equal(Buffer.from(gifHeader(512,512,false)).includes(Buffer.from('NETSCAPE2.0')),false);

// Exercise player events with a deterministic clock, without browser automation.
for (const count of [16,20]) {
const elements = Object.fromEntries(['recipe','step','speed','continuous','frame','label','sheet','play','export','speedLabel','status'].map(x=>[x,{value:'',textContent:'',checked:false}]));
elements.recipe.value='example';elements.step.value=1;elements.speed.value=8;
elements.step.max=count;
if(count===16){
 elements.sequenceMetadata={textContent:JSON.stringify({example:{bend:{mode:'sampler_mix',mix_method:'alternate',mix_first:'plms',mix_history:'shared',mix_block:1,mix_exchange:.5,mix_curve:'constant',eta:0},steps:{'1':{active_sampler:'plms',sampler_iteration:1,exchange_weight:0},'2':{active_sampler:'ddim',sampler_iteration:2,exchange_weight:0}}}})};
 elements.sequenceSettings={textContent:''};elements.stepDetails={textContent:''};
}
let pending,delay;
const sandbox={document:{getElementById:id=>elements[id]},Image:class{},setTimeout:(fn,ms)=>{pending=fn;delay=ms;return 1;},clearTimeout:()=>{pending=null;},module:{exports:{}}};
vm.runInNewContext(fs.readFileSync(require.resolve('../scripts/step_viewer.js'),'utf8'),sandbox);
if(count===16){assert.match(elements.stepDetails.textContent,/Update 1: PLMS/);assert.match(elements.sequenceSettings.textContent,/one generation seed/);}
elements.play.onclick(); assert.equal(elements.play.textContent,'Pause'); assert.equal(delay,130);
for(let i=0;i<count-1;i++) pending();
assert.equal(elements.step.value,count);assert.equal(delay,1000);
for(let i=0;i<count-1;i++) pending();
assert.equal(elements.step.value,1);assert.equal(delay,1000);
pending();assert.equal(elements.play.textContent,'Play');
elements.continuous.checked=true;elements.continuous.onchange();elements.play.onclick();
for(let i=0;i<2*count-2;i++) pending();
assert.equal(elements.play.textContent,'Pause');
pending();elements.speed.value=20;elements.speed.oninput();assert.equal(delay,50);
elements.step.value=7;elements.step.oninput();assert.equal(pending,null);assert.match(elements.frame.src,/-step7.png$/);
}
console.log('Player direction, endpoint delays, speed, one-cycle stop, continuous loop, scrubbing and GIF headers passed.');

// Optional real GIF for an independent image decoder to verify.
if(process.argv[2]) {
 const dataContext={window:{}};vm.runInNewContext(fs.readFileSync(process.argv[2],'utf8'),dataContext);
 const data=dataContext.window.stepGifData;
 const chunks=[gifHeader(data.width,data.height,true)];
 for(const frame of timeline(data.frames.length,8,true)){const source=data.frames[frame.step-1];chunks.push(gifFrame(data.width,data.height,Buffer.from(source.pixels,'base64'),Buffer.from(source.palette,'base64'),frame.delay));}
 chunks.push(Uint8Array.of(0x3b));fs.writeFileSync(process.argv[3],Buffer.concat(chunks.map(x=>Buffer.from(x))));
}

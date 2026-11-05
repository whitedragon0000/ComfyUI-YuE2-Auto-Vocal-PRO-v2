// Layout only: native ComfyUI progress is the single visible progress indicator.
// NO custom onDrawForeground, NO custom progress subscription/percentage.
import { app } from "../../scripts/app.js";

function layout(node) {
  const widgets = node.widgets || [];
  const mode = widgets.find(w => w.name === 'whisper_mode');
  const file = widgets.find(w => w.name === 'whisper_model');
  if (mode) {
    mode.label = 'Whisper mode';
    mode.options = { ...mode.options, tooltip: 'Auto, Selected model, or None. Manual reference lyrics bypass ASR.' };
  }
  if (file) {
    file.label = 'Whisper file (models/whisper)';
    file.options = { ...file.options, tooltip: 'Select a real .pt or .safetensors file in ComfyUI/models/whisper.' };
  }
  const [computedW = 590, computedH = 600] = node.computeSize?.() || [];
  const w = Math.max(610, node.size?.[0] || 0, computedW);
  const h = Math.max(680, node.size?.[1] || 0, computedH + 52);
  if (node.size?.[0] < w || node.size?.[1] < h) node.setSize?.([w, h]);
}

app.registerExtension({
  name: 'yue2.auto_vocal.reference_voice_layout',
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== 'YuE2ReferenceVoiceCover') return;
    const previousCreated = nodeType.prototype.onNodeCreated;
    const previousConfigured = nodeType.prototype.onConfigure;
    nodeType.prototype.onNodeCreated = function() {
      const result = previousCreated?.apply(this, arguments);
      layout(this);
      return result;
    };
    nodeType.prototype.onConfigure = function() {
      const result = previousConfigured?.apply(this, arguments);
      layout(this);
      return result;
    };
  },
});

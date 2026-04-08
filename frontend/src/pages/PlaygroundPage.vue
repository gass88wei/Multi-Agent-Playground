<script setup>
import { inject } from "vue";
import ChatRunner from "../components/ChatRunner.vue";
import GraphViewer from "../components/GraphViewer.vue";
import TraceViewer from "../components/TraceViewer.vue";
import { I18N_KEY } from "../i18n";

const props = defineProps({
  workflows: {
    type: Array,
    required: true,
  },
  agents: {
    type: Array,
    required: true,
  },
  selectedWorkflowId: {
    type: String,
    default: "",
  },
  selectedWorkflow: {
    type: Object,
    default: null,
  },
  selectedGraph: {
    type: Object,
    default: null,
  },
  activeNodeId: {
    type: String,
    default: "",
  },
  loading: {
    type: Boolean,
    default: false,
  },
  trace: {
    type: Array,
    default: () => [],
  },
  tracePlaying: {
    type: Boolean,
    default: false,
  },
  chatMessages: {
    type: Array,
    default: () => [],
  },
});

const emit = defineEmits(["run", "clear", "select-workflow"]);

const i18n = inject(I18N_KEY, null);
const t = i18n?.t || ((key) => key);
const workflowTypeLabel = i18n?.workflowTypeLabel || ((type) => type);
</script>

<template>
  <div class="playground-grid">
    <aside class="playground-col-left">
      <section class="glass-panel workflow-select-card">
        <label class="field-label">{{ t("workflow.selectWorkflow") }}</label>
        <select
          class="workflow-native-select"
          :value="props.selectedWorkflowId"
          @change="$emit('select-workflow', $event.target.value)"
        >
          <option v-for="workflow in props.workflows" :key="workflow.id" :value="workflow.id">
            {{ workflow.name }} - {{ workflowTypeLabel(workflow.type) }}
          </option>
        </select>
      </section>

      <GraphViewer
        :graph="props.selectedGraph"
        :active-node-id="props.activeNodeId"
        :trace="props.trace"
      />
    </aside>

    <section class="playground-col-center">
      <ChatRunner
        :selected-workflow-id="props.selectedWorkflowId"
        :selected-workflow="props.selectedWorkflow"
        :loading="props.loading"
        :messages="props.chatMessages"
        @run="$emit('run', $event)"
        @clear="$emit('clear')"
      />
    </section>

    <aside class="playground-col-right">
      <TraceViewer :trace="props.trace" :playing="props.tracePlaying" />
    </aside>
  </div>
</template>

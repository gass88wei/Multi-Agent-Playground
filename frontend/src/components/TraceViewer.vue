<script setup>
import { computed, inject, nextTick, ref, watch } from "vue";
import { Terminal } from "lucide-vue-next";
import { I18N_KEY } from "../i18n";

const props = defineProps({
  trace: {
    type: Array,
    default: () => [],
  },
  playing: {
    type: Boolean,
    default: false,
  },
});

const i18n = inject(I18N_KEY, null);
const t = i18n?.t || ((key) => key);
const locale = i18n?.locale;
const traceRef = ref(null);
const expandedMap = ref({});
const traceMode = ref("simple");
const hiddenSimpleTypes = new Set(["node_entered", "node_exited"]);

const visibleTrace = computed(() => {
  const source = Array.isArray(props.trace) ? props.trace : [];
  if (traceMode.value === "detail") return source;
  return source.filter((event) => !hiddenSimpleTypes.has(String(event?.type || "")));
});

function traceKey(event, index) {
  return `${event?.at || "no-at"}-${index}`;
}

function isCollapsible(event) {
  return event?.type === "node_entered" || event?.type === "node_exited";
}

function isExpanded(event, index) {
  if (!isCollapsible(event)) return true;
  return !!expandedMap.value[traceKey(event, index)];
}

function toggleEvent(event, index) {
  if (!isCollapsible(event)) return;
  const key = traceKey(event, index);
  expandedMap.value = {
    ...expandedMap.value,
    [key]: !expandedMap.value[key],
  };
}

function formatTime(isoString) {
  try {
    const targetLocale = locale?.value === "zh-CN" ? "zh-CN" : "en-GB";
    return new Date(isoString).toLocaleTimeString(targetLocale, {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return isoString;
  }
}

watch(
  () => visibleTrace.value.length,
  async () => {
    if (!traceRef.value) return;
    await nextTick();
    traceRef.value.scrollTop = traceRef.value.scrollHeight;
  },
);
</script>

<template>
  <section class="glass-panel trace-shell">
    <header class="run-panel-header">
      <h3 class="run-panel-title">
        <Terminal :size="18" class="text-slate-400" />
        Trace
      </h3>
      <div class="trace-toolbar">
        <div class="trace-mode-switch" role="tablist" aria-label="Trace mode">
          <button
            type="button"
            class="trace-mode-button"
            :class="{ active: traceMode === 'simple' }"
            @click="traceMode = 'simple'"
          >
            {{ t("trace.showKey") }}
          </button>
          <button
            type="button"
            class="trace-mode-button"
            :class="{ active: traceMode === 'detail' }"
            @click="traceMode = 'detail'"
          >
            {{ t("trace.showAll") }}
          </button>
        </div>
        <span class="chip chip-green">{{ playing ? "Running" : "Live" }}</span>
      </div>
    </header>

    <div v-if="visibleTrace.length" ref="traceRef" class="trace-list">
      <article
        v-for="(event, index) in visibleTrace"
        :key="`${event.at}-${index}`"
        class="trace-item"
      >
        <div class="trace-dot"></div>
        <div class="trace-body">
          <div class="trace-time">{{ formatTime(event.at) }}</div>
          <div
            class="trace-card"
            :class="{ collapsed: isCollapsible(event) && !isExpanded(event, index) }"
          >
            <button
              class="trace-head trace-head-button"
              :class="{ collapsible: isCollapsible(event) }"
              type="button"
              :aria-disabled="!isCollapsible(event)"
              @click="toggleEvent(event, index)"
            >
              <strong>{{ event.title }}</strong>
              <span class="trace-head-right">
                <span class="chip">{{ event.type }}</span>
                <span v-if="isCollapsible(event)" class="trace-expand-indicator">
                  {{ isExpanded(event, index) ? "v" : ">" }}
                </span>
              </span>
            </button>
            <template v-if="isExpanded(event, index)">
              <p>{{ event.detail }}</p>
              <pre v-if="event.payload && Object.keys(event.payload).length">{{ JSON.stringify(event.payload, null, 2) }}</pre>
            </template>
          </div>
        </div>
      </article>
    </div>
    <div v-else class="trace-empty">{{ t("trace.empty") }}</div>
  </section>
</template>

import React, { useState, useEffect, useMemo } from 'react';
import { 
  LayoutDashboard, 
  Users, 
  GitBranch, 
  Play, 
  Plus, 
  MessageSquare, 
  ChevronRight, 
  Bot, 
  Settings2, 
  Activity,
  ArrowRight,
  CheckCircle2,
  Info,
  Send,
  Terminal,
  Workflow as WorkflowIcon,
  Zap,
  BrainCircuit,
  Network,
  X,
  Globe,
  Code,
  Image as ImageIcon,
  FileText,
  Database,
  Mail,
  ShoppingBag,
  Trash2,
  Library
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// --- Types ---

type Agent = {
  id: string;
  name: string;
  role: string;
  description: string;
  color: string;
  skills: string[];
};

type WorkflowType = 'router' | 'planner' | 'handoff';

type Workflow = {
  id: string;
  name: string;
  type: WorkflowType;
  description: string;
  agentIds: string[];
  createdAt: number;
};

type TraceStep = {
  id: string;
  from: string;
  to: string;
  action: string;
  timestamp: number;
  content?: string;
};

type Message = {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  agentName?: string;
};

// --- Mock Initial Data ---

const SKILL_LIBRARY = [
  { id: 's1', name: 'Web Search', description: 'Access real-time information from the internet.', icon: 'Globe' },
  { id: 's2', name: 'Code Interpreter', description: 'Execute Python code for data analysis and math.', icon: 'Code' },
  { id: 's3', name: 'Image Generation', description: 'Create high-quality images from text prompts.', icon: 'Image' },
  { id: 's4', name: 'File Analysis', description: 'Extract and summarize content from uploaded files.', icon: 'FileText' },
  { id: 's5', name: 'Database Query', description: 'Query structured data from SQL/NoSQL databases.', icon: 'Database' },
  { id: 's6', name: 'Email Automation', description: 'Draft and send emails based on user instructions.', icon: 'Mail' },
];

const INITIAL_AGENTS: Agent[] = [
  { id: 'a1', name: 'Architecture Coach', role: 'System Architect', description: '擅长解释架构设计、模块边界和技术权衡。', color: 'bg-blue-500', skills: ['s1', 's2'] },
  { id: 'a2', name: 'Documentation Writer', role: 'Technical Writer', description: '擅长把技术思路整理成清晰、易读的说明文档。', color: 'bg-purple-500', skills: ['s4'] },
  { id: 'a3', name: 'Learning Coach', role: 'Education Specialist', description: '擅长给出学习路径、实践建议和推进步骤。', color: 'bg-emerald-500', skills: [] },
];

const INITIAL_WORKFLOWS: Workflow[] = [
  {
    id: 'w1',
    name: 'Default Router Demo',
    type: 'router',
    description: '先由 router 识别用户意图，再把请求交给最合适的 specialist agent，最后由 finalizer 统一收口。',
    agentIds: ['a1', 'a2', 'a3'],
    createdAt: Date.now(),
  }
];

// --- Components ---

const Badge = ({ children, className = "" }: { children: React.ReactNode, className?: string }) => (
  <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${className}`}>
    {children}
  </span>
);

export default function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'agents' | 'workflows' | 'playground'>('overview');
  const [agents, setAgents] = useState<Agent[]>(INITIAL_AGENTS);
  const [workflows, setWorkflows] = useState<Workflow[]>(INITIAL_WORKFLOWS);
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string>(INITIAL_WORKFLOWS[0].id);

  const selectedWorkflow = useMemo(() => 
    workflows.find(w => w.id === selectedWorkflowId) || workflows[0], 
    [workflows, selectedWorkflowId]
  );

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-slate-900 rounded-xl flex items-center justify-center text-white shadow-lg shadow-slate-200">
              <BrainCircuit size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-slate-900">Agent Playground</h1>
              <p className="text-xs text-slate-500 font-medium">Multi-Agent Orchestration & Trace System</p>
            </div>
          </div>
          
          <nav className="flex items-center gap-1 bg-slate-100 p-1 rounded-full border border-slate-200">
            <button 
              onClick={() => setActiveTab('overview')}
              className={`nav-tab ${activeTab === 'overview' ? 'nav-tab-active' : 'nav-tab-inactive'} flex items-center gap-2`}
            >
              <LayoutDashboard size={16} /> 概览
            </button>
            <button 
              onClick={() => setActiveTab('agents')}
              className={`nav-tab ${activeTab === 'agents' ? 'nav-tab-active' : 'nav-tab-inactive'} flex items-center gap-2`}
            >
              <Users size={16} /> Agents
            </button>
            <button 
              onClick={() => setActiveTab('workflows')}
              className={`nav-tab ${activeTab === 'workflows' ? 'nav-tab-active' : 'nav-tab-inactive'} flex items-center gap-2`}
            >
              <GitBranch size={16} /> Workflows
            </button>
            <button 
              onClick={() => setActiveTab('playground')}
              className={`nav-tab ${activeTab === 'playground' ? 'nav-tab-active' : 'nav-tab-inactive'} flex items-center gap-2`}
            >
              <Play size={16} /> Playground
            </button>
          </nav>

          <div className="flex items-center gap-2">
            <Badge className="bg-slate-900 text-white">MVP</Badge>
            <div className="h-8 w-px bg-slate-200 mx-2" />
            <div className="flex items-center gap-2 text-slate-600">
              <Activity size={16} className="text-emerald-500 animate-pulse" />
              <span className="text-xs font-mono">System Ready</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6">
        <AnimatePresence mode="wait">
          {activeTab === 'overview' && (
            <motion.div 
              key="overview"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <Overview agents={agents} workflows={workflows} onNavigate={setActiveTab} />
            </motion.div>
          )}
          {activeTab === 'agents' && (
            <motion.div 
              key="agents"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <AgentsList agents={agents} setAgents={setAgents} />
            </motion.div>
          )}
          {activeTab === 'workflows' && (
            <motion.div 
              key="workflows"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <WorkflowsList 
                workflows={workflows} 
                setWorkflows={setWorkflows} 
                agents={agents}
                onSelect={setSelectedWorkflowId}
                selectedId={selectedWorkflowId}
              />
            </motion.div>
          )}
          {activeTab === 'playground' && (
            <motion.div 
              key="playground"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <Playground 
                workflow={selectedWorkflow} 
                workflows={workflows}
                agents={agents}
                onSelectWorkflow={setSelectedWorkflowId}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

// --- Sub-components ---

function Overview({ agents, workflows, onNavigate }: { 
  agents: Agent[], 
  workflows: Workflow[], 
  onNavigate: (tab: any) => void 
}) {
  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass-card p-8 flex flex-col justify-between relative overflow-hidden">
          <div className="relative z-10">
            <Badge className="bg-blue-100 text-blue-700 mb-4">Workflow-First Playground</Badge>
            <h2 className="text-4xl font-bold text-slate-900 leading-tight mb-4">
              先定义协作结构，<br />再观察 agent 在工作流中的真实运行过程。
            </h2>
            <p className="text-slate-500 max-w-xl text-lg">
              这个 playground 不是单纯的聊天页，而是用来展示 agent + workflow + trace 的教学型界面。建议先配置角色，再组合工作流，最后到 Playground 里跑一轮完整交互。
            </p>
          </div>
          <div className="absolute -right-20 -bottom-20 w-80 h-80 bg-blue-50 rounded-full blur-3xl opacity-50" />
        </div>

        <div className="space-y-4">
          {[
            { label: 'Agents', count: agents.length, icon: Users, color: 'text-blue-600', bg: 'bg-blue-50' },
            { label: 'Workflows', count: workflows.length, icon: GitBranch, color: 'text-purple-600', bg: 'bg-purple-50' },
            { label: 'Templates', count: 4, icon: WorkflowIcon, color: 'text-emerald-600', bg: 'bg-emerald-50' },
          ].map((stat, i) => (
            <div key={i} className="glass-card p-6 flex items-center justify-between group hover:border-slate-300 transition-colors cursor-default">
              <div className="flex items-center gap-4">
                <div className={`w-12 h-12 ${stat.bg} rounded-xl flex items-center justify-center ${stat.color}`}>
                  <stat.icon size={24} />
                </div>
                <div>
                  <p className="text-sm font-medium text-slate-500">{stat.label}</p>
                  <p className="text-2xl font-bold text-slate-900">{stat.count}</p>
                </div>
              </div>
              <ChevronRight size={20} className="text-slate-300 group-hover:text-slate-500 transition-colors" />
            </div>
          ))}
        </div>
      </div>

      <div className="glass-card p-8">
        <h3 className="text-xl font-bold text-slate-900 mb-6 flex items-center gap-2">
          <Settings2 size={20} className="text-slate-400" /> 使用顺序
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {[
            { step: '01', title: '创建 Agents', desc: '先创建或挑选可复用的 specialist agents。', tab: 'agents' },
            { step: '02', title: '组建 Workflow', desc: '再创建一个 workflow，把这些 agents 组织成协作结构。', tab: 'workflows' },
            { step: '03', title: '进入 Playground', desc: '最后进入 Playground，对选中的 workflow 发消息并观察 graph / trace。', tab: 'playground' },
          ].map((item, i) => (
            <div 
              key={i} 
              onClick={() => onNavigate(item.tab)}
              className="group cursor-pointer space-y-4"
            >
              <div className="flex items-center gap-4">
                <span className="text-4xl font-black text-slate-100 group-hover:text-slate-200 transition-colors">{item.step}</span>
                <div className="h-px flex-1 bg-slate-100 group-hover:bg-slate-200 transition-colors" />
              </div>
              <h4 className="text-lg font-bold text-slate-900 group-hover:text-blue-600 transition-colors flex items-center gap-2">
                {item.title} <ArrowRight size={16} className="opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all" />
              </h4>
              <p className="text-slate-500 text-sm leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="glass-card p-8">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-bold text-slate-900 flex items-center gap-2">
            <Activity size={20} className="text-emerald-500" /> 当前默认 Workflow
          </h3>
          <button 
            onClick={() => onNavigate('playground')}
            className="text-sm font-medium text-blue-600 hover:text-blue-700 flex items-center gap-1"
          >
            进入 Playground 时会默认加载它 <ArrowRight size={14} />
          </button>
        </div>
        <div className="p-6 bg-slate-50 rounded-xl border border-slate-200">
          <div className="flex items-start justify-between">
            <div>
              <h4 className="font-bold text-slate-900 text-lg">{workflows[0]?.name}</h4>
              <p className="text-xs font-mono text-slate-400 mb-4">workflow_{workflows[0]?.id}</p>
              <div className="flex flex-wrap gap-2">
                <Badge className="bg-slate-200 text-slate-700">{workflows[0]?.type}</Badge>
                {workflows[0]?.agentIds.map(aid => {
                  const agent = agents.find(a => a.id === aid);
                  return (
                    <span key={aid} className="text-xs text-slate-500 flex items-center gap-1">
                      <div className={`w-1.5 h-1.5 rounded-full ${agent?.color || 'bg-slate-400'}`} />
                      {agent?.name}
                    </span>
                  );
                })}
              </div>
            </div>
            <WorkflowIcon size={32} className="text-slate-200" />
          </div>
        </div>
      </div>
    </div>
  );
}

function SkillIcon({ name, size = 16, className = "" }: { name: string, size?: number, className?: string }) {
  switch (name) {
    case 'Globe': return <Globe size={size} className={className} />;
    case 'Code': return <Code size={size} className={className} />;
    case 'Image': return <ImageIcon size={size} className={className} />;
    case 'FileText': return <FileText size={size} className={className} />;
    case 'Database': return <Database size={size} className={className} />;
    case 'Mail': return <Mail size={size} className={className} />;
    default: return <Zap size={size} className={className} />;
  }
}

function AgentsList({ agents, setAgents }: { agents: Agent[], setAgents: React.Dispatch<React.SetStateAction<Agent[]>> }) {
  const [isAdding, setIsAdding] = useState(false);
  const [editingAgent, setEditingAgent] = useState<Agent | null>(null);
  const [isStoreOpen, setIsStoreOpen] = useState(false);
  const [storeAgentId, setStoreAgentId] = useState<string | null>(null);
  const [newAgent, setNewAgent] = useState({ name: '', role: '', description: '' });

  const handleAdd = () => {
    if (!newAgent.name) return;
    const colors = ['bg-blue-500', 'bg-purple-500', 'bg-emerald-500', 'bg-orange-500', 'bg-rose-500', 'bg-indigo-500'];
    const agent: Agent = {
      id: Math.random().toString(36).substr(2, 9),
      ...newAgent,
      color: colors[agents.length % colors.length],
      skills: []
    };
    setAgents([...agents, agent]);
    setNewAgent({ name: '', role: '', description: '' });
    setIsAdding(false);
  };

  const handleUpdate = () => {
    if (!editingAgent || !editingAgent.name) return;
    setAgents(agents.map(a => a.id === editingAgent.id ? editingAgent : a));
    setEditingAgent(null);
  };

  const toggleSkill = (agentId: string, skillId: string) => {
    setAgents(prev => prev.map(a => {
      if (a.id !== agentId) return a;
      const hasSkill = a.skills.includes(skillId);
      return {
        ...a,
        skills: hasSkill 
          ? a.skills.filter(id => id !== skillId)
          : [...a.skills, skillId]
      };
    }));
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-slate-900">Agents</h2>
          <p className="text-slate-500">这里管理可复用的角色单元。一个 agent 代表一种稳定职责，而不是一次性的 workflow 节点。</p>
        </div>
        <button 
          onClick={() => setIsAdding(true)}
          className="bg-slate-900 text-white px-4 py-2 rounded-xl font-medium flex items-center gap-2 hover:bg-slate-800 transition-colors shadow-lg shadow-slate-200"
        >
          <Plus size={18} /> 创建 Agent
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {isAdding && (
          <motion.div 
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="glass-card p-6 border-2 border-dashed border-blue-200 bg-blue-50/30"
          >
            <div className="space-y-4">
              <input 
                autoFocus
                placeholder="Agent 名称"
                className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                value={newAgent.name}
                onChange={e => setNewAgent({...newAgent, name: e.target.value})}
              />
              <input 
                placeholder="角色定位 (e.g. Architect)"
                className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                value={newAgent.role}
                onChange={e => setNewAgent({...newAgent, role: e.target.value})}
              />
              <textarea 
                placeholder="职责描述..."
                className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none h-24 resize-none"
                value={newAgent.description}
                onChange={e => setNewAgent({...newAgent, description: e.target.value})}
              />
              <div className="flex gap-2">
                <button onClick={handleAdd} className="flex-1 bg-blue-600 text-white py-2 rounded-lg text-sm font-bold">保存</button>
                <button onClick={() => setIsAdding(false)} className="px-4 py-2 text-slate-500 text-sm font-medium">取消</button>
              </div>
            </div>
          </motion.div>
        )}

        {agents.map((agent) => (
          <div key={agent.id} className="glass-card p-6 group hover:border-slate-300 transition-all flex flex-col h-full">
            <div className="flex items-start justify-between mb-4">
              <div className={`w-10 h-10 ${agent.color} rounded-xl flex items-center justify-center text-white shadow-lg shadow-slate-100`}>
                <Bot size={20} />
              </div>
              <div className="flex gap-2">
                <button 
                  onClick={() => {
                    setStoreAgentId(agent.id);
                    setIsStoreOpen(true);
                  }}
                  className="p-2 rounded-lg bg-slate-50 text-slate-400 hover:bg-blue-50 hover:text-blue-600 transition-all"
                  title="Install Skills"
                >
                  <ShoppingBag size={18} />
                </button>
                <button 
                  onClick={() => setEditingAgent(agent)}
                  className="p-2 rounded-lg bg-slate-50 text-slate-400 hover:bg-slate-900 hover:text-white transition-all"
                >
                  <Settings2 size={18} />
                </button>
              </div>
            </div>
            <h3 className="font-bold text-slate-900 text-lg">{agent.name}</h3>
            <p className="text-xs font-mono text-slate-400 mb-3">agent_{agent.id}</p>
            <Badge className="bg-slate-100 text-slate-600 mb-4 inline-block">{agent.role}</Badge>
            <p className="text-sm text-slate-500 leading-relaxed mb-6 flex-1">{agent.description}</p>
            
            {agent.skills.length > 0 && (
              <div className="pt-4 border-t border-slate-100">
                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2">Installed Skills</p>
                <div className="flex flex-wrap gap-2">
                  {agent.skills.map(sid => {
                    const skill = SKILL_LIBRARY.find(s => s.id === sid);
                    return (
                      <div key={sid} className="flex items-center gap-1.5 px-2 py-1 bg-slate-50 rounded-lg border border-slate-100 group/skill">
                        <SkillIcon name={skill?.icon || ''} size={12} className="text-slate-400" />
                        <span className="text-[10px] font-bold text-slate-600">{skill?.name}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Edit Modal */}
      <AnimatePresence>
        {editingAgent && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-6">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setEditingAgent(null)}
              className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="relative w-full max-w-md bg-white rounded-2xl shadow-2xl border border-slate-200 overflow-hidden"
            >
              <div className="p-6 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
                <h3 className="font-bold text-slate-900">编辑 Agent</h3>
                <button onClick={() => setEditingAgent(null)} className="text-slate-400 hover:text-slate-600">
                  <X size={20} />
                </button>
              </div>
              <div className="p-6 space-y-4 max-h-[70vh] overflow-y-auto">
                <div className="space-y-1">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">名称</label>
                  <input 
                    className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm font-medium focus:ring-2 focus:ring-blue-500 outline-none"
                    value={editingAgent.name}
                    onChange={e => setEditingAgent({...editingAgent, name: e.target.value})}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">角色</label>
                  <input 
                    className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm font-medium focus:ring-2 focus:ring-blue-500 outline-none"
                    value={editingAgent.role}
                    onChange={e => setEditingAgent({...editingAgent, role: e.target.value})}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">描述</label>
                  <textarea 
                    rows={3}
                    className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm font-medium focus:ring-2 focus:ring-blue-500 outline-none resize-none"
                    value={editingAgent.description}
                    onChange={e => setEditingAgent({...editingAgent, description: e.target.value})}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">颜色</label>
                  <div className="flex gap-2">
                    {['bg-blue-500', 'bg-purple-500', 'bg-emerald-500', 'bg-orange-500', 'bg-rose-500', 'bg-indigo-500'].map(c => (
                      <button 
                        key={c}
                        onClick={() => setEditingAgent({...editingAgent, color: c})}
                        className={`w-6 h-6 rounded-full ${c} ring-offset-2 transition-all ${editingAgent.color === c ? 'ring-2 ring-slate-900 scale-110' : 'hover:scale-110'}`}
                      />
                    ))}
                  </div>
                </div>
                
                <div className="space-y-2 pt-2">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Skills</label>
                  <div className="space-y-2">
                    {editingAgent.skills.length === 0 ? (
                      <p className="text-xs text-slate-400 italic">No skills installed.</p>
                    ) : (
                      editingAgent.skills.map(sid => {
                        const skill = SKILL_LIBRARY.find(s => s.id === sid);
                        return (
                          <div key={sid} className="flex items-center justify-between p-2 bg-slate-50 rounded-lg border border-slate-100">
                            <div className="flex items-center gap-2">
                              <SkillIcon name={skill?.icon || ''} className="text-slate-400" />
                              <span className="text-sm font-medium text-slate-700">{skill?.name}</span>
                            </div>
                            <button 
                              onClick={() => setEditingAgent({
                                ...editingAgent,
                                skills: editingAgent.skills.filter(id => id !== sid)
                              })}
                              className="text-slate-300 hover:text-rose-500 transition-colors"
                            >
                              <Trash2 size={16} />
                            </button>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              </div>
              <div className="p-6 bg-slate-50 border-t border-slate-100 flex gap-3">
                <button 
                  onClick={() => setEditingAgent(null)}
                  className="flex-1 px-4 py-2 rounded-lg text-sm font-bold text-slate-500 hover:bg-slate-200 transition-all"
                >
                  取消
                </button>
                <button 
                  onClick={handleUpdate}
                  className="flex-1 px-4 py-2 bg-slate-900 text-white rounded-lg text-sm font-bold hover:bg-slate-800 transition-all"
                >
                  保存
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Skill Store Modal */}
      <AnimatePresence>
        {isStoreOpen && (
          <div className="fixed inset-0 z-[110] flex items-center justify-center p-6">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsStoreOpen(false)}
              className="absolute inset-0 bg-slate-900/60 backdrop-blur-md"
            />
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="relative w-full max-w-2xl bg-white rounded-3xl shadow-2xl border border-slate-200 overflow-hidden"
            >
              <div className="p-8 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-blue-600 rounded-2xl flex items-center justify-center text-white shadow-lg shadow-blue-100">
                    <Library size={24} />
                  </div>
                  <div>
                    <h3 className="text-xl font-black text-slate-900 tracking-tight uppercase">Skill Mirror</h3>
                    <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Global Skill Repository</p>
                  </div>
                </div>
                <button onClick={() => setIsStoreOpen(false)} className="p-2 rounded-xl hover:bg-slate-200 transition-all">
                  <X size={24} />
                </button>
              </div>
              
              <div className="p-8 grid grid-cols-1 md:grid-cols-2 gap-4 max-h-[60vh] overflow-y-auto">
                {SKILL_LIBRARY.map(skill => {
                  const isInstalled = agents.find(a => a.id === storeAgentId)?.skills.includes(skill.id);
                  return (
                    <div 
                      key={skill.id} 
                      className={`p-5 rounded-2xl border transition-all duration-300 flex flex-col ${
                        isInstalled 
                          ? 'bg-blue-50/50 border-blue-200 ring-1 ring-blue-100' 
                          : 'bg-white border-slate-100 hover:border-slate-300 hover:shadow-xl hover:shadow-slate-100'
                      }`}
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${isInstalled ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-500'}`}>
                          <SkillIcon name={skill.icon} size={20} />
                        </div>
                        {isInstalled && (
                          <Badge className="bg-blue-600 text-white">Installed</Badge>
                        )}
                      </div>
                      <h4 className="font-bold text-slate-900 mb-1">{skill.name}</h4>
                      <p className="text-xs text-slate-500 leading-relaxed mb-6 flex-1">{skill.description}</p>
                      <button 
                        onClick={() => storeAgentId && toggleSkill(storeAgentId, skill.id)}
                        className={`w-full py-2.5 rounded-xl text-xs font-black uppercase tracking-widest transition-all ${
                          isInstalled 
                            ? 'bg-slate-100 text-slate-400 hover:bg-rose-50 hover:text-rose-600' 
                            : 'bg-slate-900 text-white hover:bg-blue-600 shadow-lg shadow-slate-200'
                        }`}
                      >
                        {isInstalled ? 'Uninstall' : 'Install Skill'}
                      </button>
                    </div>
                  );
                })}
              </div>
              
              <div className="p-8 bg-slate-50 border-t border-slate-100 flex justify-end">
                <button 
                  onClick={() => setIsStoreOpen(false)}
                  className="px-8 py-3 bg-slate-900 text-white rounded-xl text-sm font-black uppercase tracking-widest hover:bg-slate-800 transition-all shadow-xl shadow-slate-200"
                >
                  Done
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}

function WorkflowsList({ 
  workflows, 
  setWorkflows, 
  agents, 
  onSelect, 
  selectedId 
}: { 
  workflows: Workflow[], 
  setWorkflows: React.Dispatch<React.SetStateAction<Workflow[]>>,
  agents: Agent[],
  onSelect: (id: string) => void,
  selectedId: string
}) {
  const [isAdding, setIsAdding] = useState(false);
  const [newWorkflow, setNewWorkflow] = useState({ name: '', type: 'router' as WorkflowType, description: '', agentIds: [] as string[] });

  const handleAdd = () => {
    if (!newWorkflow.name) return;
    const workflow: Workflow = {
      id: Math.random().toString(36).substr(2, 9),
      ...newWorkflow,
      createdAt: Date.now()
    };
    setWorkflows([...workflows, workflow]);
    setIsAdding(false);
  };

  const toggleAgent = (id: string) => {
    setNewWorkflow(prev => ({
      ...prev,
      agentIds: prev.agentIds.includes(id) 
        ? prev.agentIds.filter(aid => aid !== id)
        : [...prev.agentIds, id]
    }));
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-slate-900">Workflows</h2>
          <p className="text-slate-500">这里定义协作模式。当前支持 router_specialists，后续可以继续接入 planner 和 handoff。</p>
        </div>
        <button 
          onClick={() => setIsAdding(true)}
          className="bg-slate-900 text-white px-4 py-2 rounded-xl font-medium flex items-center gap-2 hover:bg-slate-800 transition-colors shadow-lg shadow-slate-200"
        >
          <Plus size={18} /> 创建 Workflow
        </button>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {isAdding && (
          <motion.div 
            initial={{ scale: 0.98, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="glass-card p-8 border-2 border-dashed border-purple-200 bg-purple-50/30"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="space-y-4">
                <h4 className="font-bold text-slate-900">基本信息</h4>
                <input 
                  autoFocus
                  placeholder="Workflow 名称"
                  className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-purple-500 outline-none"
                  value={newWorkflow.name}
                  onChange={e => setNewWorkflow({...newWorkflow, name: e.target.value})}
                />
                <select 
                  className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-purple-500 outline-none"
                  value={newWorkflow.type}
                  onChange={e => setNewWorkflow({...newWorkflow, type: e.target.value as WorkflowType})}
                >
                  <option value="router">Router Specialists (路由模式)</option>
                  <option value="planner">Planner (计划模式)</option>
                  <option value="handoff">Dynamic Delegation (动态委派)</option>
                </select>
                <textarea 
                  placeholder="协作逻辑描述..."
                  className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-purple-500 outline-none h-24 resize-none"
                  value={newWorkflow.description}
                  onChange={e => setNewWorkflow({...newWorkflow, description: e.target.value})}
                />
              </div>
              <div className="space-y-4">
                <h4 className="font-bold text-slate-900">绑定 Agents ({newWorkflow.agentIds.length})</h4>
                <div className="grid grid-cols-1 gap-2 max-h-48 overflow-y-auto pr-2">
                  {agents.map(agent => (
                    <label key={agent.id} className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-all ${newWorkflow.agentIds.includes(agent.id) ? 'bg-purple-100 border-purple-300' : 'bg-white border-slate-200 hover:border-slate-300'}`}>
                      <div className="flex items-center gap-3">
                        <div className={`w-2 h-2 rounded-full ${agent.color}`} />
                        <span className="text-sm font-medium text-slate-700">{agent.name}</span>
                      </div>
                      <input 
                        type="checkbox" 
                        className="hidden"
                        checked={newWorkflow.agentIds.includes(agent.id)}
                        onChange={() => toggleAgent(agent.id)}
                      />
                      {newWorkflow.agentIds.includes(agent.id) && <CheckCircle2 size={16} className="text-purple-600" />}
                    </label>
                  ))}
                </div>
                <div className="flex gap-2 pt-4">
                  <button onClick={handleAdd} className="flex-1 bg-purple-600 text-white py-2 rounded-lg text-sm font-bold">保存工作流</button>
                  <button onClick={() => setIsAdding(false)} className="px-4 py-2 text-slate-500 text-sm font-medium">取消</button>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {workflows.map((workflow) => (
          <div 
            key={workflow.id} 
            onClick={() => onSelect(workflow.id)}
            className={`glass-card p-6 cursor-pointer transition-all group ${selectedId === workflow.id ? 'ring-2 ring-slate-900 border-transparent' : 'hover:border-slate-300'}`}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <h3 className="font-bold text-slate-900 text-xl">{workflow.name}</h3>
                  <Badge className="bg-slate-900 text-white">{workflow.type}</Badge>
                </div>
                <p className="text-xs font-mono text-slate-400 mb-4">workflow_{workflow.id}</p>
                <p className="text-sm text-slate-500 mb-6 max-w-2xl">{workflow.description}</p>
                
                <div className="flex items-center gap-4">
                  <div className="flex -space-x-2">
                    {workflow.agentIds.map((aid, i) => {
                      const agent = agents.find(a => a.id === aid);
                      return (
                        <div 
                          key={aid} 
                          title={agent?.name}
                          className={`w-8 h-8 rounded-full border-2 border-white ${agent?.color || 'bg-slate-200'} flex items-center justify-center text-white text-[10px] font-bold`}
                          style={{ zIndex: 10 - i }}
                        >
                          {agent?.name.charAt(0)}
                        </div>
                      );
                    })}
                  </div>
                  <span className="text-xs text-slate-400 font-medium">
                    {workflow.agentIds.length} Agents involved
                  </span>
                </div>
              </div>
              <div className={`p-4 rounded-2xl transition-colors ${selectedId === workflow.id ? 'bg-slate-900 text-white' : 'bg-slate-50 text-slate-300 group-hover:text-slate-500'}`}>
                <WorkflowIcon size={24} />
              </div>
            </div>
            {selectedId === workflow.id && (
              <div className="mt-4 pt-4 border-t border-slate-100 flex items-center justify-between">
                <span className="text-xs text-emerald-600 font-bold flex items-center gap-1">
                  <CheckCircle2 size={12} /> 当前已选中
                </span>
                <button className="text-xs font-bold text-slate-900 hover:underline">编辑配置</button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function Playground({ 
  workflow, 
  workflows, 
  agents, 
  onSelectWorkflow 
}: { 
  workflow: Workflow, 
  workflows: Workflow[],
  agents: Agent[],
  onSelectWorkflow: (id: string) => void
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [trace, setTrace] = useState<TraceStep[]>([]);

  const handleSend = async () => {
    if (!input.trim() || isRunning) return;
    
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsRunning(true);
    setTrace([]);

    // Simulate multi-agent execution
    const steps: TraceStep[] = [];
    const addStep = (from: string, to: string, action: string, content?: string) => {
      const step = { id: Math.random().toString(), from, to, action, timestamp: Date.now(), content };
      steps.push(step);
      setTrace([...steps]);
    };

    await new Promise(r => setTimeout(r, 600));
    addStep('START', 'Router', '识别意图');
    
    await new Promise(r => setTimeout(r, 1000));
    // Pick a random agent from workflow
    const targetAgentId = workflow.agentIds[Math.floor(Math.random() * workflow.agentIds.length)];
    const targetAgent = agents.find(a => a.id === targetAgentId);
    const agentName = targetAgent?.name || 'Specialist';
    addStep('Router', agentName, '路由请求', `用户询问关于 ${input.substring(0, 10)}...`);

    await new Promise(r => setTimeout(r, 1200));
    addStep(agentName, 'Finalizer', '生成回复', '已完成专业领域分析');

    await new Promise(r => setTimeout(r, 600));
    addStep('Finalizer', 'END', '输出结果');

    const assistantMsg: Message = { 
      id: (Date.now() + 1).toString(), 
      role: 'assistant', 
      agentName: targetAgent?.name,
      content: `[${targetAgent?.name}] 针对您的问题，我从${targetAgent?.role}的角度进行了分析。这是一个基于 ${workflow.name} 工作流生成的模拟回复。`
    };
    setMessages(prev => [...prev, assistantMsg]);
    setIsRunning(false);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-180px)]">
      {/* Left: Config & Graph */}
      <div className="lg:col-span-4 flex flex-col gap-6 overflow-hidden">
        <div className="glass-card p-6 shrink-0">
          <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">选择 Workflow</label>
          <select 
            className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5 text-sm font-bold text-slate-900 focus:ring-2 focus:ring-slate-900 outline-none appearance-none cursor-pointer"
            value={workflow.id}
            onChange={e => onSelectWorkflow(e.target.value)}
          >
            {workflows.map(w => (
              <option key={w.id} value={w.id}>{w.name} • {w.type}</option>
            ))}
          </select>
        </div>

        <div className="glass-card flex-1 flex flex-col overflow-hidden">
          <div className="p-4 border-b border-slate-100 flex items-center justify-between">
            <h3 className="font-bold text-slate-900 flex items-center gap-2">
              <Network size={18} className="text-blue-500" /> Workflow Graph
            </h3>
            <span className="text-[10px] font-bold text-slate-400 uppercase">可视化拓扑</span>
          </div>
          
          <div className="flex-1 relative overflow-hidden bg-slate-50/50 p-4">
            <WorkflowVisualGraph 
              workflow={workflow} 
              agents={agents} 
              trace={trace} 
            />
          </div>

          <div className="p-4 bg-slate-900 text-slate-400 font-mono text-[10px] overflow-hidden whitespace-nowrap">
            {trace.length > 0 ? (
              <div className="animate-in fade-in slide-in-from-bottom-2">
                <span className="text-emerald-400">●</span> {trace[trace.length-1].from} → {trace[trace.length-1].to} ({trace[trace.length-1].action})
              </div>
            ) : '> System idle. Waiting for input...'}
          </div>
        </div>
      </div>

      {/* Center: Chat */}
      <div className="lg:col-span-5 flex flex-col glass-card overflow-hidden">
        <div className="p-4 border-b border-slate-100 flex items-center justify-between bg-white">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-emerald-100 text-emerald-600 rounded-lg flex items-center justify-center">
              <MessageSquare size={18} />
            </div>
            <div>
              <h3 className="font-bold text-slate-900 text-sm">Chat Runner</h3>
              <p className="text-[10px] text-slate-400 font-medium">Active: {workflow.name}</p>
            </div>
          </div>
          <button 
            onClick={() => setMessages([])}
            className="text-xs font-bold text-slate-400 hover:text-slate-600"
          >
            Clear
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-slate-50/30">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center space-y-4 opacity-40">
              <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center text-slate-400">
                <Send size={32} />
              </div>
              <div>
                <p className="font-bold text-slate-900">开始测试工作流</p>
                <p className="text-sm">发送一条消息，观察 Agent 之间的协作</p>
              </div>
            </div>
          )}
          {messages.map((m) => (
            <div key={m.id} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[85%] space-y-1 ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
                {m.agentName && (
                  <span className="text-[10px] font-bold text-slate-400 uppercase ml-1">{m.agentName}</span>
                )}
                <div className={`p-4 rounded-2xl text-sm leading-relaxed shadow-sm ${
                  m.role === 'user' 
                    ? 'bg-slate-900 text-white rounded-tr-none' 
                    : 'bg-white border border-slate-200 text-slate-700 rounded-tl-none'
                }`}>
                  {m.content}
                </div>
              </div>
            </div>
          ))}
          {isRunning && (
            <div className="flex justify-start">
              <div className="bg-white border border-slate-200 p-4 rounded-2xl rounded-tl-none shadow-sm flex items-center gap-2">
                <div className="flex gap-1">
                  <div className="w-1.5 h-1.5 bg-slate-300 rounded-full animate-bounce" />
                  <div className="w-1.5 h-1.5 bg-slate-300 rounded-full animate-bounce [animation-delay:0.2s]" />
                  <div className="w-1.5 h-1.5 bg-slate-300 rounded-full animate-bounce [animation-delay:0.4s]" />
                </div>
                <span className="text-xs font-medium text-slate-400">Thinking...</span>
              </div>
            </div>
          )}
        </div>

        <div className="p-4 bg-white border-t border-slate-100">
          <div className="relative">
            <input 
              placeholder="输入测试指令..."
              className="w-full bg-slate-100 border-none rounded-xl pl-4 pr-12 py-3 text-sm focus:ring-2 focus:ring-slate-900 outline-none"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSend()}
            />
            <button 
              onClick={handleSend}
              disabled={!input.trim() || isRunning}
              className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 bg-slate-900 text-white rounded-lg flex items-center justify-center hover:bg-slate-800 disabled:opacity-50 transition-all"
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Right: Trace */}
      <div className="lg:col-span-3 flex flex-col glass-card overflow-hidden">
        <div className="p-4 border-b border-slate-100 flex items-center justify-between bg-white">
          <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
            <Terminal size={18} className="text-slate-400" /> Trace
          </h3>
          <Badge className="bg-emerald-100 text-emerald-700">Live</Badge>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/50">
          {trace.length === 0 && (
            <div className="h-full flex items-center justify-center text-center p-8">
              <p className="text-xs text-slate-400 font-medium leading-relaxed">
                运行 workflow 后显示 trace 事件。<br />这里将记录节点切换与决策轨迹。
              </p>
            </div>
          )}
          {trace.map((step, i) => (
            <motion.div 
              key={step.id}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className="relative pl-6 border-l-2 border-slate-200 py-1"
            >
              <div className="absolute left-[-5px] top-2 w-2 h-2 rounded-full bg-blue-500 ring-4 ring-blue-50" />
              <div className="text-[10px] font-mono text-slate-400 mb-1">
                {new Date(step.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
              </div>
              <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-sm space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-900">{step.action}</span>
                  <Badge className="bg-slate-100 text-slate-500 lowercase">{step.from} → {step.to}</Badge>
                </div>
                {step.content && (
                  <p className="text-[10px] text-slate-500 bg-slate-50 p-2 rounded-lg font-mono leading-tight">
                    {step.content}
                  </p>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}

// --- Visual Graph Component ---

function WorkflowVisualGraph({ workflow, agents, trace }: { 
  workflow: Workflow, 
  agents: Agent[], 
  trace: TraceStep[] 
}) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const workflowAgents = workflow.agentIds.map(id => agents.find(a => a.id === id)).filter(Boolean) as Agent[];
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  
  type GraphNode = {
    id: string;
    label: string;
    x: number; // Pixels
    y: number; // Pixels
    type: 'terminal' | 'logic' | 'agent';
    icon?: React.ReactNode;
  };

  const [nodePositions, setNodePositions] = useState<GraphNode[]>([]);

  // Update dimensions and initial positions
  useEffect(() => {
    if (!containerRef.current) return;
    
    const updateLayout = () => {
      const rect = containerRef.current!.getBoundingClientRect();
      setDimensions({ width: rect.width, height: rect.height });

      const w = rect.width;
      const h = rect.height;

      const initialNodes: GraphNode[] = [
        { id: 'START', label: 'Start Session', x: w / 2, y: h * 0.1, type: 'terminal' },
        { id: 'Router', label: 'Intention Router', x: w / 2, y: h * 0.28, type: 'logic', icon: <BrainCircuit size={16} /> },
        ...workflowAgents.map((a, i) => ({
          id: a.name,
          label: a.name,
          x: workflowAgents.length > 1 
            ? (w * 0.15) + ((w * 0.7) / (workflowAgents.length - 1)) * i 
            : w / 2,
          y: h * 0.52,
          type: 'agent' as const,
          icon: <Bot size={16} />
        })),
        { id: 'Finalizer', label: 'Response Finalizer', x: w / 2, y: h * 0.78, type: 'logic', icon: <Zap size={16} /> },
        { id: 'END', label: 'End Session', x: w / 2, y: h * 0.92, type: 'terminal' }
      ];
      setNodePositions(initialNodes);
    };

    updateLayout();
    const observer = new ResizeObserver(updateLayout);
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [workflow.id, agents]);

  const handleDrag = (id: string, info: any) => {
    setNodePositions(prev => prev.map(n => {
      if (n.id === id) {
        return { ...n, x: n.x + info.delta.x, y: n.y + info.delta.y };
      }
      return n;
    }));
  };

  const connections = [
    { from: 'START', to: 'Router' },
    ...workflowAgents.map(a => ({ from: 'Router', to: a.name })),
    ...workflowAgents.map(a => ({ from: a.name, to: 'Finalizer' })),
    { from: 'Finalizer', to: 'END' }
  ];

  return (
    <div ref={containerRef} className="w-full h-full relative bg-slate-50/50 rounded-xl overflow-hidden border border-slate-100 select-none">
      <div className="absolute inset-0 opacity-[0.03]" style={{ backgroundImage: 'radial-gradient(#000 1px, transparent 0)', backgroundSize: '24px 24px' }} />
      
      <svg className="w-full h-full absolute inset-0 pointer-events-none overflow-visible">
        {connections.map((conn, i) => {
          const fromNode = nodePositions.find(n => n.id === conn.from);
          const toNode = nodePositions.find(n => n.id === conn.to);
          if (!fromNode || !toNode) return null;

          const isActive = trace.some(s => s.from === conn.from && s.to === conn.to);
          
          // Improved Bezier Logic: Use dynamic control points for better entry angles
          const dy = toNode.y - fromNode.y;
          const cp1x = fromNode.x;
          const cp1y = fromNode.y + dy * 0.4;
          const cp2x = toNode.x;
          const cp2y = toNode.y - dy * 0.4;

          // Calculate the exact tangent angle at the end of the curve (P3 - P2)
          const angle = Math.atan2(toNode.y - cp2y, toNode.x - cp2x);
          const radius = 26; // Node radius + small buffer
          
          // The line and arrow tip should both stop at the node boundary
          const endX = toNode.x - Math.cos(angle) * radius;
          const endY = toNode.y - Math.sin(angle) * radius;
          
          // Start the line from the boundary of the source node
          const startAngle = Math.atan2(cp1y - fromNode.y, cp1x - fromNode.x);
          const startX = fromNode.x + Math.cos(startAngle) * radius;
          const startY = fromNode.y + Math.sin(startAngle) * radius;

          const d = `M ${startX} ${startY} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${endX} ${endY}`;
          
          // Manual arrow calculation at the exact end point
          const arrowSize = 10;
          const xA = endX - arrowSize * Math.cos(angle - 0.4);
          const yA = endY - arrowSize * Math.sin(angle - 0.4);
          const xB = endX - arrowSize * Math.cos(angle + 0.4);
          const yB = endY - arrowSize * Math.sin(angle + 0.4);
          const arrowD = `M ${endX} ${endY} L ${xA} ${yA} L ${xB} ${yB} Z`;

          return (
            <g key={i}>
              {/* Base Path */}
              <path d={d} fill="none" stroke="#cbd5e1" strokeWidth="2" strokeDasharray={isActive ? "none" : "4 4"} />
              <path d={arrowD} fill="#cbd5e1" />
              
              {/* Active Path */}
              {isActive && (
                <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <motion.path 
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    d={d} fill="none" stroke="#3b82f6" strokeWidth="3"
                    transition={{ duration: 0.5 }}
                  />
                  <path d={arrowD} fill="#3b82f6" />
                </motion.g>
              )}
            </g>
          );
        })}
      </svg>

      {nodePositions.map((node) => {
        const isActive = trace.some(s => s.from === node.id || s.to === node.id);
        const isCurrent = trace.length > 0 && trace[trace.length - 1].to === node.id;
        const isHovered = hoveredNode === node.id;
        
        return (
          <motion.div 
            key={node.id}
            drag
            dragMomentum={false}
            onDrag={(_, info) => handleDrag(node.id, info)}
            className="absolute -translate-x-1/2 -translate-y-1/2 z-10 cursor-grab active:cursor-grabbing"
            style={{ x: node.x, y: node.y }}
            onMouseEnter={() => setHoveredNode(node.id)}
            onMouseLeave={() => setHoveredNode(null)}
          >
            {/* Tooltip */}
            <AnimatePresence>
              {isHovered && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.8, y: 10 }}
                  animate={{ opacity: 1, scale: 1, y: -45 }}
                  exit={{ opacity: 0, scale: 0.8, y: 10 }}
                  className="absolute left-1/2 -translate-x-1/2 px-3 py-1.5 bg-slate-900 text-white text-[10px] font-bold rounded-lg whitespace-nowrap shadow-xl pointer-events-none z-20"
                >
                  {node.label}
                  <div className="absolute bottom-[-4px] left-1/2 -translate-x-1/2 w-2 h-2 bg-slate-900 rotate-45" />
                </motion.div>
              )}
            </AnimatePresence>

            {/* Node Circle */}
            <motion.div 
              animate={{ 
                scale: isCurrent ? 1.2 : isHovered ? 1.1 : 1,
                backgroundColor: isCurrent ? '#3b82f6' : isActive ? '#ffffff' : '#f8fafc',
                borderColor: isCurrent ? '#2563eb' : isActive ? '#3b82f6' : '#e2e8f0',
              }}
              className={`w-12 h-12 rounded-full border-2 flex items-center justify-center shadow-lg relative transition-colors`}
            >
              {node.type === 'terminal' ? (
                <div className={`w-3 h-3 rounded-full ${isCurrent ? 'bg-white' : isActive ? 'bg-blue-500' : 'bg-slate-400'}`} />
              ) : (
                <div className={isCurrent ? 'text-white' : isActive ? 'text-blue-500' : 'text-slate-500'}>
                  {node.icon}
                </div>
              )}

              {isCurrent && (
                <motion.div 
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0, 0.5] }}
                  transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
                  className="absolute inset-[-4px] border-2 border-blue-400 rounded-full pointer-events-none"
                />
              )}
            </motion.div>
          </motion.div>
        );
      })}
    </div>
  );
}

import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import streamlit.components.v1 as components

from normalizer.engine import NormalizationEngine
from analyzer.engine import AnalyzerEngine
from ai.bob_client import BobClient, BobAPIError
from ai.fraud_ai import FraudAIAnalyzer, FRAUD_TYPES
from utils.ids import safe_case_id


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="FraudGraph",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
CASES_DIR = BASE_DIR / "data" / "cases"
CASES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SIMPLE STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container { max-width: 1600px; padding-top: 1rem; padding-left: 1.15rem; padding-right: 1.15rem; }
    .step { color: #6b7280; font-size: 0.85rem; margin-bottom: 0.25rem; }
    .hint { color: #6b7280; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "page": "case",
    "case_id": "",
    "case_title": "",
    "case_meta": {},
    "result": None,
    "analysis": None,
    "ai_analysis": None,
    "raw_files": [],
    "normalized_json": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_app():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def go(page: str):
    st.session_state.page = page
    st.rerun()


def write_case_file(case_id: str, result: dict):
    case_dir = CASES_DIR / case_id
    original_dir = case_dir / "original"
    normalized_dir = case_dir / "normalized"
    original_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    for uploaded_file in st.session_state.raw_files:
        target = original_dir / Path(uploaded_file.name).name
        target.write_bytes(uploaded_file.getvalue())

    output_file = normalized_dir / "case.json"
    output_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_file


def build_phase2_export(analysis: dict):
    entity_edges = analysis.get("network", {}).get("entity_network", {}).get("edges", [])
    return {
        "analysis": analysis.get("analysis", {}),
        "subject_profiles": analysis.get("subject_profiles", []),
        "entity_relationships": entity_edges,
        "findings": analysis.get("findings", []),
        "data_quality": analysis.get("data_quality", {}),
    }


def write_analysis_file(case_id: str, analysis: dict):
    analysis_dir = CASES_DIR / case_id / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_file = analysis_dir / "case_analysis.json"
    output_file.write_text(
        json.dumps(build_phase2_export(analysis), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_file


def _event_search_blob(event: dict) -> str:
    try:
        return json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(event)


def build_visualizer_payload(analysis: dict):
    """Build an interactive index over the completed analyzer output."""
    profiles = analysis.get("subject_profiles", [])
    edges = analysis.get("entity_relationships")
    if edges is None:
        edges = analysis.get("network", {}).get("entity_network", {}).get("edges", [])

    profile_by_id = {p.get("subject_id"): p for p in profiles if p.get("subject_id")}
    name_by_subject = {p.get("subject_id"): (p.get("name") or p.get("subject_id")) for p in profiles if p.get("subject_id")}

    # Human-readable ownership labels for technical entities. This is presentation metadata only;
    # it does not change the analyzed relationships or event evidence.
    identifier_type_map = {
        "account_ids": "ACCOUNT", "account_numbers": "ACCOUNT",
        "phone_ids": "PHONE", "phone_numbers": "PHONE",
        "device_ids": "DEVICE", "imeis": "IMEI", "observed_imeis": "IMEI",
        "sim_ids": "SIM", "imsis": "SIM", "observed_imsis": "SIM",
        "ip_ids": "IP", "ip_addresses": "IP",
    }
    owner_names = {}
    for profile in profiles:
        owner = profile.get("name") or profile.get("subject_id") or "Unknown person"
        for field, entity_type in identifier_type_map.items():
            for value in (profile.get("identifiers", {}).get(field) or []):
                if value is not None and str(value).strip():
                    owner_names.setdefault((entity_type, str(value)), set()).add(owner)

    event_by_id = {}
    event_evidence = {}
    all_events = []
    for profile in profiles:
        for event in profile.get("timeline", []):
            event_id = event.get("event_id") or event.get("details", {}).get("call_id") or event.get("details", {}).get("transaction_id")
            if not event_id:
                event_id = f"{profile.get('subject_id','unknown')}::{event.get('timestamp','')}::{event.get('type','')}"
            if event_id not in event_by_id:
                clean = dict(event)
                clean["_subject_id"] = profile.get("subject_id")
                clean["_subject_name"] = profile.get("name") or profile.get("subject_id")
                event_by_id[event_id] = clean
                all_events.append(clean)
            for eid in event.get("evidence_ids", []) or event.get("details", {}).get("source_evidence_ids", []):
                event_evidence.setdefault(eid, set()).add(event_id)

    nodes = {}
    type_order = ["PERSON", "ACCOUNT", "PHONE", "DEVICE", "SIM", "IMEI", "IP"]

    def add_node(node_id, node_type):
        if not node_id:
            return
        key = f"{node_type}:{node_id}"
        if key in nodes:
            return
        linked_names = sorted(owner_names.get((node_type, str(node_id)), set()))
        if node_type == "PERSON":
            label = name_by_subject.get(node_id, node_id)
            graph_label = label
        elif len(linked_names) == 1:
            label = node_id
            graph_label = f"{linked_names[0]}\n{node_id}"
        elif linked_names:
            label = node_id
            graph_label = f"{len(linked_names)} linked people\n{node_id}"
        else:
            label = node_id
            graph_label = node_id
        nodes[key] = {
            "id": key, "entity_id": node_id, "type": node_type, "label": label,
            "graph_label": graph_label, "owner_names": linked_names,
            "display": f"{label} · {node_id}" if label != node_id else node_id,
        }

    for profile in profiles:
        add_node(profile.get("subject_id"), "PERSON")

    normalized_edges = []
    for idx, edge in enumerate(edges):
        source, target = edge.get("source", ""), edge.get("target", "")
        source_type, target_type = edge.get("source_type", "UNKNOWN"), edge.get("target_type", "UNKNOWN")
        if not source or not target:
            continue
        add_node(source, source_type)
        add_node(target, target_type)
        normalized_edges.append({
            "id": f"E{idx+1}", "source": f"{source_type}:{source}", "target": f"{target_type}:{target}",
            "source_entity": source, "target_entity": target, "source_type": source_type, "target_type": target_type,
            "relationship": edge.get("relationship_type", "RELATED"), "event_count": edge.get("event_count", 0),
            "evidence_ids": edge.get("evidence_ids", []) or [],
        })

    def events_for_evidence(evidence_ids):
        ids = set()
        for evidence_id in evidence_ids:
            ids.update(event_evidence.get(evidence_id, set()))
        return sorted((event_by_id[eid] for eid in ids if eid in event_by_id), key=lambda x: x.get("timestamp", ""), reverse=True)

    node_events = {}
    for key, node in nodes.items():
        entity_id, node_type = node["entity_id"], node["type"]
        if node_type == "PERSON":
            profile = profile_by_id.get(entity_id, {})
            events = []
            seen = set()
            for event in profile.get("timeline", []):
                eid = event.get("event_id")
                if eid in seen:
                    continue
                clean = dict(event)
                clean["_subject_id"] = profile.get("subject_id")
                clean["_subject_name"] = profile.get("name") or profile.get("subject_id")
                events.append(clean)
                seen.add(eid)
            node_events[key] = sorted(events, key=lambda x: x.get("timestamp", ""), reverse=True)
        else:
            matches, seen, token = [], set(), str(entity_id)
            for event in all_events:
                eid = event.get("event_id")
                if eid in seen:
                    continue
                if token and token in _event_search_blob(event):
                    matches.append(event)
                    seen.add(eid)
            node_events[key] = sorted(matches, key=lambda x: x.get("timestamp", ""), reverse=True)

    edge_details = {}
    for edge in normalized_edges:
        evs = events_for_evidence(edge["evidence_ids"])
        edge_details[edge["id"]] = {
            "calls": [e for e in evs if str(e.get("type", "")).upper() == "CALL"],
            "transactions": [e for e in evs if str(e.get("type", "")).upper() in {"TRANSACTION", "TRANSFER"}],
            "other": [e for e in evs if str(e.get("type", "")).upper() not in {"CALL", "TRANSACTION", "TRANSFER"}],
        }

    counts = {
        "nodes": len(nodes), "edges": len(normalized_edges),
        "calls": sum(1 for e in all_events if str(e.get("type", "")).upper() == "CALL"),
        "transactions": sum(1 for e in all_events if str(e.get("type", "")).upper() in {"TRANSACTION", "TRANSFER"}),
    }
    return {
        "nodes": list(nodes.values()), "edges": normalized_edges, "node_events": node_events, "edge_details": edge_details,
        "counts": counts, "relationship_types": sorted({e["relationship"] for e in normalized_edges}),
        "entity_types": sorted({n["type"] for n in nodes.values()}, key=lambda x: (type_order.index(x) if x in type_order else 99, x)),
    }


def render_investigation_visualizer(analysis: dict):
    payload = build_visualizer_payload(analysis)
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
:root{
  --ink:#0f172a;--muted:#64748b;--line:#dfe6ef;--line-strong:#cbd5e1;--soft:#f6f8fb;--white:#ffffff;
  --blue:#2563eb;--blue-soft:#eff6ff;--green:#059669;--red:#dc2626;--amber:#d97706;--violet:#7c3aed;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:#fff;color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow:hidden}
body{min-height:100%}
button,input,select{font:inherit}
.fg-shell{width:100%;height:100%;border:1px solid var(--line);background:var(--white);border-radius:20px;overflow:hidden;box-shadow:0 16px 48px rgba(15,23,42,.08)}
.fg-head{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:16px 20px 14px;border-bottom:1px solid var(--line);background:linear-gradient(180deg,#fff 0,#fbfcfe 100%)}
.fg-brand{display:flex;align-items:center;gap:12px;min-width:0}.fg-mark{width:38px;height:38px;border-radius:11px;background:var(--blue-soft);border:1px solid #bfdbfe;color:var(--blue);display:grid;place-items:center;font-weight:850;font-size:15px;box-shadow:inset 0 0 0 5px rgba(37,99,235,.035)}
.fg-title{font-size:18px;font-weight:820;letter-spacing:-.03em;line-height:1.05}.fg-sub{font-size:11px;color:var(--muted);margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fg-status{display:flex;align-items:center;gap:8px;padding:7px 10px;border:1px solid var(--line);border-radius:10px;background:#fff;font-size:10px;font-weight:760;color:#475569;white-space:nowrap}.status-dot{width:8px;height:8px;border-radius:999px;background:#10b981;box-shadow:0 0 0 4px rgba(16,185,129,.1)}
.fg-metrics{display:grid;grid-template-columns:repeat(4,minmax(88px,1fr));gap:8px;padding:10px 14px;border-bottom:1px solid var(--line);background:#fff}.m-card{border:1px solid var(--line);border-radius:11px;background:#fff;padding:8px 10px;min-width:0}.m-k{font-size:18px;font-weight:830;letter-spacing:-.025em}.m-l{font-size:9px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);font-weight:760;margin-top:2px}
.fg-tools{display:grid;grid-template-columns:minmax(210px,1.5fr) repeat(3,minmax(130px,1fr)) auto;gap:8px;padding:10px 14px;border-bottom:1px solid var(--line);background:#fff}.fg-control{height:38px;padding:0 11px;border:1px solid var(--line-strong);border-radius:10px;background:#fff;color:var(--ink);font-size:11px;outline:none;min-width:0}.fg-control:focus{border-color:#93c5fd;box-shadow:0 0 0 3px #dbeafe}.fg-btn{height:38px;padding:0 12px;border:1px solid var(--line-strong);border-radius:10px;background:#fff;color:var(--ink);font-size:11px;font-weight:780;cursor:pointer;white-space:nowrap}.fg-btn:hover{background:#f8fafc}.fg-btn.primary{background:var(--blue);border-color:var(--blue);color:#fff}.fg-btn.primary:hover{filter:brightness(.97)}
.fg-workspace{display:grid;grid-template-columns:minmax(0,1fr) 430px;height:760px;min-height:0}.fg-graph-pane{position:relative;min-width:0;border-right:1px solid var(--line);background:radial-gradient(circle at 50% 46%,#fff 0,#fbfdff 58%,#f6f8fb 100%)}#network{width:100%;height:760px}.graph-hint{position:absolute;right:14px;top:14px;padding:7px 9px;background:rgba(255,255,255,.93);border:1px solid var(--line);border-radius:9px;font-size:9px;color:#64748b;box-shadow:0 6px 18px rgba(15,23,42,.05)}
.legend{position:absolute;left:14px;bottom:14px;max-width:92%;display:flex;gap:8px;flex-wrap:wrap;padding:9px 11px;background:rgba(255,255,255,.96);border:1px solid var(--line);border-radius:11px;box-shadow:0 8px 22px rgba(15,23,42,.05)}.legend-item{font-size:9px;color:#475569;display:flex;align-items:center;gap:5px;font-weight:700}.dot{width:9px;height:9px;border-radius:50%;border:1px solid rgba(15,23,42,.12);display:inline-block}
.fg-panel{min-width:0;display:flex;flex-direction:column;background:#fff}.p-head{padding:17px 17px 13px;border-bottom:1px solid var(--line);min-height:126px}.eyebrow{font-size:9px;letter-spacing:.11em;text-transform:uppercase;color:#64748b;font-weight:820}.p-title{font-size:19px;font-weight:830;letter-spacing:-.025em;margin-top:5px;word-break:break-word}.p-id{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#64748b;margin-top:4px;word-break:break-all}.p-pills{display:flex;gap:6px;flex-wrap:wrap;margin-top:9px}.pill{font-size:9px;padding:4px 7px;border-radius:999px;background:#f8fafc;color:#475569;border:1px solid #e2e8f0;font-weight:720}
.p-body{padding:13px 15px 10px;min-height:0;display:flex;flex-direction:column;flex:1}.p-section{min-height:0;flex:1;display:flex;flex-direction:column}.section-head{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px}.section-title{font-size:11px;font-weight:820}.section-note{font-size:9px;color:#94a3b8;font-weight:700}.event-list{display:grid;gap:7px;min-height:0}.event-card{border:1px solid var(--line);border-radius:11px;background:#fff;padding:9px 10px;min-height:0}.event-row{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}.event-badge{font-size:8px;font-weight:850;letter-spacing:.08em;text-transform:uppercase;padding:4px 6px;border-radius:6px}.badge-call{background:#ecfeff;color:#0e7490}.badge-tx{background:#f5f3ff;color:#6d28d9}.badge-other{background:#f8fafc;color:#475569}.event-time{font-size:9px;color:#64748b;white-space:nowrap}.event-main{font-size:10px;line-height:1.4;margin-top:5px;color:#172033}.event-foot{display:flex;gap:5px;flex-wrap:wrap;margin-top:6px}.kv{font-size:8px;color:#475569;background:#f8fafc;border:1px solid #e5e7eb;border-radius:6px;padding:3px 5px}.evidence{font:8px ui-monospace,SFMono-Regular,Menlo,monospace;color:#64748b;word-break:break-all}
.detail-banner{display:flex;justify-content:space-between;gap:8px;align-items:center;padding:8px 9px;border:1px solid var(--line);border-radius:9px;background:#f8fafc;margin-bottom:9px}.banner-main{font-size:9px;color:#475569;line-height:1.35}.banner-main b{color:#111827}.edge-pill{font-size:9px;font-weight:800;color:#334155;white-space:nowrap}
.log-list{display:grid;gap:7px;min-height:0}.log{border:1px solid var(--line);border-radius:10px;padding:8px 9px;background:#fff}.log-grid{display:grid;grid-template-columns:1fr auto;gap:5px 10px}.log-label{font-size:8px;color:#64748b;text-transform:uppercase;letter-spacing:.06em;font-weight:760}.log-value{font-size:10px;font-weight:720;word-break:break-word}.log-value.amount{font-size:13px;color:#0f172a}.log-bottom{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}.tx-ok{color:#047857}.tx-fail{color:#b91c1c}.empty{padding:24px 14px;border:1px dashed #d8e1eb;border-radius:11px;background:#fbfcfe;color:#64748b;text-align:center;font-size:10px;line-height:1.5}
.pager{display:flex;justify-content:space-between;align-items:center;gap:8px;padding-top:10px;border-top:1px solid var(--line);margin-top:10px}.pager-count{font-size:9px;color:#64748b;font-weight:750}.pager-actions{display:flex;gap:6px}.pager-btn{height:30px;min-width:66px;padding:0 9px;border:1px solid var(--line-strong);background:#fff;border-radius:8px;font-size:9px;font-weight:780;color:#334155;cursor:pointer}.pager-btn:disabled{opacity:.42;cursor:default}.mode-tabs{display:flex;gap:5px;margin-bottom:9px}.mode-tab{height:30px;padding:0 9px;border:1px solid var(--line);border-radius:8px;background:#fff;font-size:9px;font-weight:800;color:#64748b;cursor:pointer}.mode-tab.active{background:var(--blue-soft);border-color:#bfdbfe;color:#1d4ed8}
@media(max-width:980px){.fg-workspace{grid-template-columns:1fr;height:auto}.fg-graph-pane{border-right:0;border-bottom:1px solid var(--line)}#network{height:620px}.fg-panel{height:620px}.fg-tools{grid-template-columns:1fr 1fr}.fg-status{display:none}}
@media(max-width:600px){.fg-metrics{grid-template-columns:1fr 1fr}.fg-tools{grid-template-columns:1fr}.fg-head{padding:13px 14px}.fg-sub{white-space:normal}.fg-workspace{height:auto}.fg-graph-pane,#network{height:520px}.fg-panel{height:560px}}
</style>
</head>
<body>
<div class="fg-shell" id="root">
  <div class="fg-head">
    <div class="fg-brand">
      <div class="fg-mark">FG</div>
      <div style="min-width:0">
        <div class="fg-title">Final Evidence Workspace</div>
        <div class="fg-sub">Entity intelligence graph · evidence-linked calls & transactions · exact chronological history</div>
      </div>
    </div>
    <div class="fg-status"><span class="status-dot"></span> Analysis locked · visualization only</div>
  </div>
  <div class="fg-metrics" id="metrics"></div>
  <div class="fg-tools">
    <input id="search" class="fg-control" type="search" placeholder="Search name, phone, account, device, IP…" aria-label="Search entity">
    <select id="typeFilter" class="fg-control" aria-label="Entity type"><option value="ALL">All entity types</option></select>
    <select id="relFilter" class="fg-control" aria-label="Relationship type"><option value="ALL">All relationships</option></select>
    <select id="eventFilter" class="fg-control" aria-label="Evidence event type"><option value="ALL">All evidence</option><option value="CALL">Calls</option><option value="TRANSACTION">Transactions</option></select>
    <button id="fit" class="fg-btn primary" type="button">Fit graph</button>
    <button id="reset" class="fg-btn" type="button">Reset</button>
  </div>
  <div class="fg-workspace">
    <section class="fg-graph-pane" aria-label="Entity relationship graph">
      <div id="network"></div>
      <div class="graph-hint">Click entity = history · click edge = evidence · double-click = focus</div>
      <div class="legend" id="legend"></div>
    </section>
    <aside class="fg-panel" aria-live="polite">
      <div class="p-head" id="panelHead">
        <div class="eyebrow">Investigation graph</div>
        <div class="p-title">Select an entity or relationship</div>
        <div class="p-id">Every detail shown here comes from the completed analyzer output.</div>
      </div>
      <div class="p-body" id="panelBody">
        <div class="empty">Select a named entity to open its exact timeline, or select an edge to inspect the call logs and transactions that produced the relationship.</div>
      </div>
    </aside>
  </div>
</div>
<script>(function(){
'use strict';
const DATA=__PAYLOAD__;
const COLORS={PERSON:'#2563eb',ACCOUNT:'#7c3aed',PHONE:'#0891b2',DEVICE:'#ea580c',SIM:'#16a34a',IMEI:'#64748b',IP:'#be123c',UNKNOWN:'#94a3b8'};
const EDGE_COLORS={CALLED:'#0e7490',TRANSFERRED_TO:'#7c3aed',REGISTERED_ON:'#ea580c',ASSOCIATED_WITH:'#16a34a',USED_BY:'#2563eb',RELATED:'#64748b'};
const pageSize=6;
const root=document.getElementById('root'),metrics=document.getElementById('metrics'),legend=document.getElementById('legend'),networkEl=document.getElementById('network'),panelHead=document.getElementById('panelHead'),panelBody=document.getElementById('panelBody');
const search=document.getElementById('search'),typeFilter=document.getElementById('typeFilter'),relFilter=document.getElementById('relFilter'),eventFilter=document.getElementById('eventFilter'),fitBtn=document.getElementById('fit'),resetBtn=document.getElementById('reset');
let network=null,selectedNode=null,selectedEdge=null,edgeMode='CALL',page=0;
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function fmtDate(v){if(!v)return'Unknown time';const d=new Date(v);return Number.isNaN(d.getTime())?String(v):d.toLocaleString(undefined,{year:'numeric',month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit'});}
function short(v,n){const s=String(v??'');return s.length>(n||84)?s.slice(0,(n||84)-1)+'…':s;}
function typeColor(t){return COLORS[t]||COLORS.UNKNOWN;}
function edgeColor(t){return EDGE_COLORS[t]||EDGE_COLORS.RELATED;}
function kind(e){const t=String(e?.type||'').toUpperCase();if(t==='CALL')return'CALL';if(t==='TRANSACTION'||t==='TRANSFER')return'TRANSACTION';return'OTHER';}
function details(e){return e?.details||{};}
function nameOf(id){const n=DATA.nodes.find(x=>x.id===id);return n?.label||n?.entity_id||id;}
function pageCount(total){return Math.max(1,Math.ceil(total/pageSize));}
function slicePage(items){const start=page*pageSize;return items.slice(start,start+pageSize);}
function pageFooter(total,onChange){const pages=pageCount(total);const clamped=Math.min(page,pages-1);page=clamped;return '<div class="pager"><div class="pager-count">Page '+(page+1)+' of '+pages+' · '+total+' item'+(total===1?'':'s')+'</div><div class="pager-actions"><button class="pager-btn" id="prevPage" '+(page<=0?'disabled':'')+'>Previous</button><button class="pager-btn" id="nextPage" '+(page>=pages-1?'disabled':'')+'>Next</button></div></div>';}
function bindPager(renderFn,total){const p=panelBody.querySelector('#prevPage'),n=panelBody.querySelector('#nextPage');if(p)p.addEventListener('click',()=>{page=Math.max(0,page-1);renderFn();});if(n)n.addEventListener('click',()=>{page=Math.min(pageCount(total)-1,page+1);renderFn();});}
function renderMetrics(){metrics.innerHTML='';[['Nodes',DATA.counts.nodes],['Relationships',DATA.counts.edges],['Call logs',DATA.counts.calls],['Transactions',DATA.counts.transactions]].forEach(x=>{metrics.insertAdjacentHTML('beforeend','<div class="m-card"><div class="m-k">'+esc(x[1])+'</div><div class="m-l">'+esc(x[0])+'</div></div>');});}
function initFilters(){DATA.entity_types.forEach(t=>typeFilter.insertAdjacentHTML('beforeend','<option value="'+esc(t)+'">'+esc(t)+'</option>'));DATA.relationship_types.forEach(t=>relFilter.insertAdjacentHTML('beforeend','<option value="'+esc(t)+'">'+esc(t)+'</option>'));legend.innerHTML=DATA.entity_types.map(t=>'<span class="legend-item"><i class="dot" style="background:'+typeColor(t)+'"></i>'+esc(t)+'</span>').join('');}
function matchesNode(n){if(typeFilter.value!=='ALL'&&n.type!==typeFilter.value)return false;const q=search.value.trim().toLowerCase();if(!q)return true;return [n.entity_id,n.label,n.display,n.type,n.owner_names?.join(' ')].some(v=>String(v||'').toLowerCase().includes(q));}
function matchesEdge(e){if(relFilter.value!=='ALL'&&e.relationship!==relFilter.value)return false;const f=eventFilter.value;if(f==='ALL')return true;const d=DATA.edge_details[e.id]||{};return f==='CALL'?Boolean(d.calls?.length):Boolean(d.transactions?.length);}
function buildGraph(){
  if(!window.vis){networkEl.innerHTML='<div class="empty" style="margin:20px">Graph engine could not load. The evidence tables remain available above.</div>';return;}
  const visibleNodes=DATA.nodes.filter(matchesNode),allowed=new Set(visibleNodes.map(n=>n.id));
  const visibleEdges=DATA.edges.filter(e=>allowed.has(e.source)&&allowed.has(e.target)&&matchesEdge(e));
  const ns=visibleNodes.map(n=>({
    id:n.id,
    label:short(n.graph_label||n.label,28),
    title:'<b>'+esc(n.label)+'</b><br>'+esc(n.type)+'<br><code>'+esc(n.entity_id)+'</code>'+(n.owner_names?.length?'<br>Linked name(s): '+esc(n.owner_names.join(', ')):'')+'<br>Click for timeline',
    color:{background:typeColor(n.type),border:'#ffffff',highlight:{background:'#111827',border:'#ffffff'},hover:{background:typeColor(n.type),border:'#0f172a'}},
    font:{color:'#0f172a',size:n.type==='PERSON'?16:10,face:'Inter, Arial',strokeWidth:4,strokeColor:'#ffffff',bold:n.type==='PERSON'},
    shape:'dot',size:n.type==='PERSON'?26:16,borderWidth:3
  }));
  const es=visibleEdges.map(e=>({
    id:e.id,from:e.source,to:e.target,label:e.event_count?String(e.event_count):'',title:esc(e.relationship)+' · '+e.event_count+' linked event(s)',width:Math.min(1.8+Math.log2((e.event_count||1)+1),7),
    color:{color:edgeColor(e.relationship),highlight:'#111827',hover:'#334155',opacity:0.68},
    font:{size:9,color:'#475569',strokeWidth:4,strokeColor:'#ffffff',face:'Inter, Arial'},
    smooth:{type:'dynamic'}
  }));
  if(network)network.destroy();
  network=new vis.Network(networkEl,{nodes:new vis.DataSet(ns),edges:new vis.DataSet(es)},{
    physics:{enabled:true,solver:'forceAtlas2Based',forceAtlas2Based:{gravitationalConstant:-72,centralGravity:.012,springLength:170,springConstant:.035,damping:.78,avoidOverlap:1.2},stabilization:{iterations:520,fit:true}},
    interaction:{hover:true,tooltipDelay:80,navigationButtons:true,keyboard:{enabled:true},multiselect:false,hideEdgesOnDrag:false,zoomView:true,dragView:true},
    nodes:{chosen:true},edges:{selectionWidth:3}
  });
  network.on('click',p=>{if(p.nodes?.length){selectedNode=p.nodes[0];selectedEdge=null;page=0;renderNode(selectedNode);network.selectNodes([selectedNode]);}else if(p.edges?.length){selectedEdge=p.edges[0];selectedNode=null;page=0;renderEdge(selectedEdge);network.selectEdges([selectedEdge]);}else clearSelection();});
  network.on('doubleClick',p=>{if(p.nodes?.length)network.focus(p.nodes[0],{scale:1.55,animation:{duration:450,easingFunction:'easeInOutQuad'}});});
}
function clearSelection(){selectedNode=null;selectedEdge=null;page=0;panelHead.innerHTML='<div class="eyebrow">Investigation graph</div><div class="p-title">Select an entity or relationship</div><div class="p-id">Every detail shown here comes from the completed analyzer output.</div>';panelBody.innerHTML='<div class="empty">Select a named entity to open its exact timeline, or select an edge to inspect the call logs and transactions that produced the relationship.</div>';if(network)network.unselectAll();}
function eventCard(e){const d=details(e),k=kind(e);let main='';if(k==='CALL')main=(d.caller_phone_id||'Unknown caller')+' → '+(d.receiver_phone_id||'Unknown receiver')+(d.duration_seconds!=null?' · '+d.duration_seconds+' sec':'');else if(k==='TRANSACTION')main=(d.sender_account_id||'Unknown sender')+' → '+(d.receiver_account_id||'Unknown receiver')+(d.amount!=null?' · ₹'+Number(d.amount).toLocaleString('en-IN'):'');else main=short(JSON.stringify(d),190);const badge=k==='CALL'?'badge-call':k==='TRANSACTION'?'badge-tx':'badge-other';const ev=(e.evidence_ids||d.source_evidence_ids||[]).join(', ');return '<div class="event-card"><div class="event-row"><span class="event-badge '+badge+'">'+esc(k==='OTHER'?e.type:k)+'</span><span class="event-time">'+esc(fmtDate(e.timestamp))+'</span></div><div class="event-main">'+esc(main)+'</div><div class="event-foot"><span class="kv">Event '+esc(e.event_id||'—')+'</span>'+(e.direction?'<span class="kv">'+esc(e.direction)+'</span>':'')+(ev?'<span class="evidence">'+esc(ev)+'</span>':'')+'</div></div>';}
function renderNode(id){
  const n=DATA.nodes.find(x=>x.id===id);if(!n)return;const events=(DATA.node_events[id]||[]).slice().sort((a,b)=>String(b.timestamp||'').localeCompare(String(a.timestamp||''))),calls=events.filter(e=>kind(e)==='CALL').length,tx=events.filter(e=>kind(e)==='TRANSACTION').length;
  panelHead.innerHTML='<div class="eyebrow">Entity · '+esc(n.type)+'</div><div class="p-title">'+esc(n.label)+'</div><div class="p-id">'+esc(n.entity_id)+'</div><div class="p-pills"><span class="pill">'+events.length+' timeline events</span><span class="pill">'+calls+' calls</span><span class="pill">'+tx+' transactions</span>'+(n.owner_names?.length?'<span class="pill">Linked: '+esc(n.owner_names.join(', '))+'</span>':'')+'</div>';
  const draw=()=>{const rows=slicePage(events);let html='<div class="p-section"><div class="section-head"><div class="section-title">Exact chronology</div><div class="section-note">newest → oldest</div></div><div class="event-list">'+(rows.length?rows.map(eventCard).join(''):'<div class="empty">No timeline event in the analyzer output contains this entity.</div>')+'</div>'+pageFooter(events.length)+'</div>';panelBody.innerHTML=html;bindPager(draw,events.length);};
  draw();
}
function logCard(e,type){const d=details(e);if(type==='CALL'){return '<div class="log"><div class="log-grid"><div><div class="log-label">Time</div><div class="log-value">'+esc(fmtDate(e.timestamp))+'</div></div><div><div class="log-label">Duration</div><div class="log-value">'+esc(d.duration_seconds!=null?d.duration_seconds+' sec':'—')+'</div></div><div><div class="log-label">Caller</div><div class="log-value">'+esc(d.caller_phone_id||'—')+'</div></div><div><div class="log-label">Receiver</div><div class="log-value">'+esc(d.receiver_phone_id||'—')+'</div></div></div><div class="log-bottom"><span class="kv">'+esc(e.event_id||'—')+'</span><span class="evidence">'+esc((e.evidence_ids||[]).join(', '))+'</span></div></div>';}
  const cls=String(d.status||'').toUpperCase()==='SUCCESS'?'tx-ok':String(d.status||'').toUpperCase()==='FAILED'?'tx-fail':'';return '<div class="log"><div class="log-grid"><div><div class="log-label">Time</div><div class="log-value">'+esc(fmtDate(e.timestamp))+'</div></div><div><div class="log-label">Amount</div><div class="log-value amount">'+esc(d.amount!=null?'₹'+Number(d.amount).toLocaleString('en-IN'):'—')+'</div></div><div><div class="log-label">Sender account</div><div class="log-value">'+esc(d.sender_account_id||'—')+'</div></div><div><div class="log-label">Receiver account</div><div class="log-value">'+esc(d.receiver_account_id||'—')+'</div></div></div><div class="log-bottom"><span class="kv '+cls+'">'+esc(d.status||'—')+'</span><span class="kv">'+esc(e.event_id||'—')+'</span><span class="evidence">'+esc((e.evidence_ids||[]).join(', '))+'</span></div></div>';}
function renderEdge(id){
  const e=DATA.edges.find(x=>x.id===id),d=DATA.edge_details[id]||{calls:[],transactions:[],other:[]};if(!e)return;const a=DATA.nodes.find(n=>n.id===e.source),b=DATA.nodes.find(n=>n.id===e.target);
  panelHead.innerHTML='<div class="eyebrow">Relationship · undirected</div><div class="p-title">'+esc(e.relationship)+'</div><div class="p-id">'+esc(a?.label||e.source)+' ↔ '+esc(b?.label||e.target)+'</div><div class="p-pills"><span class="pill">'+esc(e.event_count)+' linked events</span><span class="pill">'+esc(e.evidence_ids.length)+' evidence IDs</span></div>';
  const current=edgeMode==='CALL'?d.calls:d.transactions;
  const draw=()=>{const rows=slicePage(current);let html='<div class="p-section"><div class="detail-banner"><div class="banner-main">The graph is <b>undirected for navigation</b>. Direction contained in the underlying call/transaction event remains unchanged.</div><div class="edge-pill">'+esc(e.relationship)+'</div></div><div class="mode-tabs"><button class="mode-tab '+(edgeMode==='CALL'?'active':'')+'" data-mode="CALL" type="button">Call logs · '+d.calls.length+'</button><button class="mode-tab '+(edgeMode==='TX'?'active':'')+'" data-mode="TX" type="button">Transactions · '+d.transactions.length+'</button></div><div class="log-list">'+(rows.length?rows.map(x=>logCard(x,edgeMode==='CALL'?'CALL':'TRANSACTION')).join(''):'<div class="empty">No '+(edgeMode==='CALL'?'call':'transaction')+' evidence is linked to this relationship by evidence ID.</div>')+'</div>'+pageFooter(current.length)+'</div>';
    if(d.other.length)html+='<div style="font-size:9px;color:#64748b;margin-top:8px">Other evidence linked to this edge: '+d.other.length+'</div>';
    panelBody.innerHTML=html;
    panelBody.querySelectorAll('[data-mode]').forEach(btn=>btn.addEventListener('click',()=>{edgeMode=btn.dataset.mode==='CALL'?'CALL':'TX';page=0;renderEdge(id);}));
    bindPager(draw,current.length);
  };draw();
}
function apply(){buildGraph();if(selectedNode)renderNode(selectedNode);else if(selectedEdge)renderEdge(selectedEdge);}
search.addEventListener('input',()=>{page=0;apply();});typeFilter.addEventListener('change',()=>{page=0;apply();});relFilter.addEventListener('change',()=>{page=0;apply();});eventFilter.addEventListener('change',()=>{page=0;apply();});fitBtn.addEventListener('click',()=>{if(network)network.fit({animation:{duration:350,easingFunction:'easeInOutQuad'}});});resetBtn.addEventListener('click',()=>{search.value='';typeFilter.value='ALL';relFilter.value='ALL';eventFilter.value='ALL';edgeMode='CALL';clearSelection();buildGraph();});
renderMetrics();initFilters();buildGraph();
})();</script>
</body></html>"""
    components.html(html.replace("__PAYLOAD__", payload_json), height=940, scrolling=False)


def event_to_row(event: dict):
    details = event.get("details") or {}
    return {
        "timestamp": event.get("timestamp", ""),
        "type": event.get("type", ""),
        "direction": event.get("direction", ""),
        "amount_inr": details.get("amount", ""),
        "source": (
            details.get("sender_account_id")
            or details.get("caller_phone_id")
            or details.get("sender")
            or details.get("phone_id")
            or ""
        ),
        "target": (
            details.get("receiver_account_id")
            or details.get("receiver_phone_id")
            or details.get("receiver")
            or ""
        ),
        "reference": details.get("reference_number", ""),
        "evidence_ids": ", ".join(event.get("evidence_ids", [])),
    }


def build_entity_relationship_rows(analysis: dict):
    rows = []
    entity_network = analysis.get("network", {}).get("entity_network", {})
    edges = entity_network.get("edges", [])

    profile_names = {
        p.get("subject_id"): p.get("name") or p.get("subject_id")
        for p in analysis.get("subject_profiles", [])
    }

    for edge in edges:
        source_type = edge.get("source_type", "")
        target_type = edge.get("target_type", "")
        source = edge.get("source", "")
        target = edge.get("target", "")

        rows.append(
            {
                "Source Type": source_type,
                "Source": profile_names.get(source, source),
                "Source ID": source,
                "Relationship": edge.get("relationship_type", ""),
                "Target Type": target_type,
                "Target": profile_names.get(target, target),
                "Target ID": target,
                "Events": edge.get("event_count", 0),
                "Evidence IDs": ", ".join(edge.get("evidence_ids", [])),
            }
        )

    return rows


# ============================================================
# HEADER
# ============================================================

st.title("FraudGraph")
st.caption("Cyber-fraud evidence normalization and entity analysis")

page_titles = {
    "case": "Step 1 — Create Case",
    "raw": "Step 2 — Raw Data & Normalize",
    "analyzer": "Step 3 — Analyzer",
    "results": "Step 4 — Analysis Result",
}

st.markdown(
    f'<div class="step">{page_titles[st.session_state.page]}</div>',
    unsafe_allow_html=True,
)

# Small reset action, intentionally unobtrusive.
with st.sidebar:
    st.button("Start new case", on_click=reset_app, use_container_width=True)


# ============================================================
# STEP 1 — CREATE CASE
# ============================================================

if st.session_state.page == "case":
    st.header("Create a new case")
    st.write("Enter the case details first. Raw evidence is uploaded in the next step.")

    col1, col2 = st.columns(2)
    with col1:
        case_id = st.text_input(
            "Case ID *",
            placeholder="JH-CYB-001",
            value=st.session_state.case_id,
        )
        case_title = st.text_input(
            "Case name / title *",
            placeholder="Jamtara Cyber Fraud Investigation",
            value=st.session_state.case_title,
        )
        fir_number = st.text_input("FIR number", placeholder="Optional")
        police_station = st.text_input("Police station", placeholder="Optional")

    with col2:
        incident_type = st.text_input(
            "Incident type",
            placeholder="UPI Fraud / SIM Swap / Cyber Fraud",
        )
        complaint_date = st.text_input(
            "Complaint date",
            placeholder="YYYY-MM-DD",
        )
        reported_loss = st.number_input(
            "Reported loss (INR)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
        )
        description = st.text_area(
            "Case description",
            placeholder="Short description of the case",
            height=120,
        )

    if st.button("Create case and continue", type="primary", use_container_width=True):
        clean_id = safe_case_id(case_id)
        if not clean_id:
            st.error("Case ID is required.")
        elif not case_title.strip():
            st.error("Case name / title is required.")
        else:
            st.session_state.case_id = clean_id
            st.session_state.case_title = case_title.strip()
            st.session_state.case_meta = {
                "fir_number": fir_number.strip(),
                "police_station": police_station.strip(),
                "incident_type": incident_type.strip(),
                "complaint_date": complaint_date.strip(),
                "reported_loss_inr": reported_loss,
                "description": description.strip(),
            }
            st.session_state.result = None
            st.session_state.analysis = None
            st.session_state.ai_analysis = None
            st.session_state.raw_files = []
            st.session_state.normalized_json = None
            go("raw")


# ============================================================
# STEP 2 — RAW DATA + NORMALIZE
# ============================================================

elif st.session_state.page == "raw":
    st.header("Raw data")
    st.write(
        f"Case: **{st.session_state.case_title}**  ·  ID: **{st.session_state.case_id}**"
    )

    files = st.file_uploader(
        "Upload raw evidence files",
        type=["csv", "xlsx", "xls", "json", "txt", "pdf"],
        accept_multiple_files=True,
        key=f"raw_evidence_upload_{st.session_state.case_id}",
    )

    if files:
        st.caption(f"{len(files)} file(s) selected")

    if st.button("Normalize raw data", type="primary", use_container_width=True):
        if not files:
            st.error("Upload at least one raw evidence file.")
            st.stop()

        engine = NormalizationEngine(
            case_id=st.session_state.case_id,
            title=st.session_state.case_title,
        )

        meta = st.session_state.case_meta
        case_obj = engine.case["case"]
        case_obj["fir_number"] = meta.get("fir_number", "")
        case_obj["police_station"] = meta.get("police_station", "")
        case_obj["incident_type"] = meta.get("incident_type", "")
        case_obj["complaint_date"] = meta.get("complaint_date", "")
        case_obj["reported_loss_inr"] = meta.get("reported_loss_inr")
        case_obj["description"] = meta.get("description", "")

        progress = st.progress(0)
        for index, uploaded_file in enumerate(files):
            engine.process_file(
                filename=uploaded_file.name,
                file_bytes=uploaded_file.getvalue(),
            )
            progress.progress(int(((index + 1) / len(files)) * 100))

        result = engine.finalize()
        st.session_state.raw_files = list(files)
        st.session_state.result = result
        st.session_state.normalized_json = json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        write_case_file(st.session_state.case_id, result)
        go("analyzer")

    if st.session_state.result is None:
        if st.button("← Back to case", use_container_width=True):
            go("case")


# ============================================================
# STEP 3 — ANALYZER ACTIONS
# ============================================================

elif st.session_state.page == "analyzer":
    st.header("Analyzer")
    st.write("Normalization is complete. Choose what to do with the normalized case.")

    result = st.session_state.result
    counts = {
        "Persons": len(result.get("persons", [])),
        "Accounts": len(result.get("accounts", [])),
        "Devices": len(result.get("devices", [])),
        "Transactions": len(result.get("transactions", [])),
        "Calls": len(result.get("calls", [])),
        "SIM events": len(result.get("sim_events", [])),
    }

    st.dataframe(
        [{"Entity": k, "Count": v} for k, v in counts.items()],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.write("### Continue")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Analyse normalized case", type="primary", use_container_width=True):
            try:
                analysis = AnalyzerEngine(result).analyze()
                st.session_state.analysis = analysis
                write_analysis_file(st.session_state.case_id, analysis)
                go("results")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    with c2:
        st.download_button(
            "Download normalized case.json",
            data=st.session_state.normalized_json or json.dumps(result, indent=2),
            file_name=f'{st.session_state.case_id}.json',
            mime="application/json",
            use_container_width=True,
        )

    if st.button("← Back to raw data", use_container_width=True):
        go("raw")


# ============================================================
# STEP 4 — ANALYSIS RESULT
# ============================================================

else:
    analysis = st.session_state.analysis

    if not analysis:
        st.warning("No analysis is loaded.")
        if st.button("Go to analyzer", use_container_width=True):
            go("analyzer")
        st.stop()

    st.header("Analysis result")
    st.write(
        f"Case: **{analysis['analysis'].get('case_title', '')}**  ·  "
        f"ID: **{analysis['analysis'].get('case_id', '')}**"
    )

    # First option is deliberately the entity-relationship table.
    tabs = st.tabs([
        "Entity Relationships",
        "Person Profiles",
        "Timeline",
        "Graph — Phase 3",
        "AI Fraud Analysis",
    ])

    # --------------------------------------------------------
    # 4A ENTITY RELATIONSHIPS
    # --------------------------------------------------------
    with tabs[0]:
        st.subheader("Entity relationship table")
        st.caption(
            "This is the first visualization-independent analysis output. "
            "Each row represents a relationship derived from the normalized evidence."
        )

        rows = build_entity_relationship_rows(analysis)
        if rows:
            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
                height=560,
            )
        else:
            st.info("No entity relationships could be derived from the supplied evidence.")

    # --------------------------------------------------------
    # 4B PERSON PROFILES
    # --------------------------------------------------------
    with tabs[1]:
        st.subheader("Person profiles")
        profiles = analysis.get("subject_profiles", [])

        if not profiles:
            st.info("No person profiles were derived.")
        else:
            profile_rows = []
            for p in profiles:
                ids = p.get("identifiers", {})
                stats = p.get("statistics", {})
                profile_rows.append(
                    {
                        "Name": p.get("name") or "Unknown",
                        "Person ID": p.get("subject_id", ""),
                        "Role": p.get("role") or "UNSPECIFIED",
                        "Accounts": len(ids.get("account_ids", [])) or len(ids.get("account_numbers", [])),
                        "Phones": len(ids.get("phone_ids", [])) or len(ids.get("phone_numbers", [])),
                        "Devices": len(ids.get("device_ids", [])) or len(ids.get("imeis", [])),
                        "Transactions": stats.get("transactions_count", 0),
                        "Calls": stats.get("calls_count", 0),
                    }
                )

            st.dataframe(
                profile_rows,
                use_container_width=True,
                hide_index=True,
                height=500,
            )

            selected_id = st.selectbox(
                "Open profile",
                [p.get("subject_id", "") for p in profiles],
                format_func=lambda sid: next(
                    (p.get("name") or sid for p in profiles if p.get("subject_id") == sid),
                    sid,
                ),
            )

            profile = next(p for p in profiles if p.get("subject_id") == selected_id)
            st.write(f"**Name:** {profile.get('name') or 'Unknown'}")
            st.write(f"**Role:** {profile.get('role') or 'UNSPECIFIED'}")
            st.write("**Identifiers**")
            st.json(profile.get("identifiers", {}), expanded=False)

    # --------------------------------------------------------
    # 4C TIMELINE
    # --------------------------------------------------------
    with tabs[2]:
        st.subheader("Unified timeline")
        profiles = analysis.get("subject_profiles", [])
        options = [p.get("subject_id", "") for p in profiles]

        if options:
            selected_id = st.selectbox(
                "Select person",
                options,
                format_func=lambda sid: next(
                    (p.get("name") or sid for p in profiles if p.get("subject_id") == sid),
                    sid,
                ),
                key="timeline_subject",
            )
            profile = next(p for p in profiles if p.get("subject_id") == selected_id)
            timeline_rows = [event_to_row(event) for event in profile.get("timeline", [])]
            st.dataframe(
                timeline_rows,
                use_container_width=True,
                hide_index=True,
                height=560,
            )
        else:
            st.info("No person timelines are available.")

    # --------------------------------------------------------
    # 4D GRAPH — FINAL INVESTIGATION VISUALIZER
    # --------------------------------------------------------
    with tabs[3]:
        st.subheader("Final investigation visualizer")
        st.caption(
            "The graph is a navigation layer over the completed analysis: entities are nodes, "
            "relationships are undirected edges, and every interactive detail is resolved back to "
            "the analyzer's evidence-linked events."
        )
        render_investigation_visualizer(analysis)

    # --------------------------------------------------------
    # 4E AI FRAUD ANALYSIS — IBM BOB
    # --------------------------------------------------------
    with tabs[4]:
        st.subheader("AI fraud analysis")
        st.caption(
            "IBM Bob analyzes the derived Phase 2 case only. It does not replace the evidence engine "
            "and it cannot establish guilt. Role outputs are investigative hypotheses."
        )

        with st.expander("IBM Bob connection", expanded=st.session_state.ai_analysis is None):
            env_key = os.getenv("BOB_API_KEY") or os.getenv("BOBSHELL_API_KEY")
            api_key = st.text_input(
                "Bob API key",
                type="password",
                value="" if env_key else "",
                help="Prefer an Inference API key. If you use a General key, also provide Team ID.",
            )
            team_id = st.text_input(
                "Team ID (only needed for a General key)",
                value=os.getenv("BOB_TEAM_ID", ""),
            )
            base_url = st.text_input(
                "Bob inference base URL",
                value=os.getenv("BOB_API_BASE_URL", "https://api.us-east.bob.ibm.com/inference/v1"),
            )
            model = st.text_input(
                "Bob model",
                value=os.getenv("BOB_MODEL", "premium"),
                help="Use the model alias enabled for your IBM Bob subscription/hackathon credits.",
            )

        selected_key = api_key.strip() or os.getenv("BOB_API_KEY") or os.getenv("BOBSHELL_API_KEY") or ""
        if not selected_key:
            st.info("Add BOB_API_KEY in your environment or enter the key above to enable live AI inference.")
        else:
            st.success("IBM Bob is configured for inference.")

        if st.button("Run AI Fraud Analysis", type="primary", use_container_width=True):
            if not selected_key:
                st.error("IBM Bob API key is required.")
            else:
                try:
                    client = BobClient(
                        api_key=selected_key,
                        team_id=team_id.strip() or None,
                        base_url=base_url.strip(),
                        model=model.strip() or "premium",
                    )
                    with st.spinner("IBM Bob is analyzing the derived case..."):
                        ai_result = FraudAIAnalyzer(client).analyze(analysis)
                    st.session_state.ai_analysis = ai_result
                    st.success("AI analysis completed.")
                except BobAPIError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"AI analysis failed: {exc}")

        ai_result = st.session_state.ai_analysis
        if ai_result:
            prediction = ai_result.get("fraud_type_prediction", {})
            roles = ai_result.get("role_analysis", {})

            st.divider()
            st.write("### Predicted fraud type")
            st.metric(
                "Primary fraud type",
                prediction.get("primary_type", "Unknown"),
                f"{prediction.get('estimated_probability_pct', 0):.1f}% estimated probability",
            )

            distribution = prediction.get("distribution", [])
            if distribution:
                st.dataframe(
                    [
                        {
                            "Fraud Type": row.get("fraud_type", ""),
                            "Estimated Probability (%)": row.get("estimated_probability_pct", 0),
                            "AI Reasoning": " | ".join(row.get("reasons", [])),
                        }
                        for row in distribution
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            if prediction.get("explanation"):
                st.write("**AI explanation:**", prediction["explanation"])

            st.divider()
            st.write("### Suspect-role analysis")
            role_rows = []
            for key, label in [
                ("prime_suspect", "Prime suspect"),
                ("kingpin_candidate", "Kingpin candidate"),
            ]:
                cand = roles.get(key) or {}
                role_rows.append({
                    "Role": label,
                    "Subject ID": cand.get("subject_id") or "—",
                    "Name": cand.get("name") or "—",
                    "Confidence (%)": cand.get("confidence_pct", 0),
                    "Reasons": " | ".join(cand.get("reasons", [])),
                })
            for cand in roles.get("mule_candidates", [])[:5]:
                role_rows.append({
                    "Role": "Mule candidate",
                    "Subject ID": cand.get("subject_id") or "—",
                    "Name": cand.get("name") or "—",
                    "Confidence (%)": cand.get("confidence_pct", 0),
                    "Reasons": " | ".join(cand.get("reasons", [])),
                })
            st.dataframe(role_rows, use_container_width=True, hide_index=True)

            st.divider()
            st.write("### Evidence cross-checks")
            for item in ai_result.get("cross_checks", []):
                st.write("•", item)
            if ai_result.get("limitations"):
                st.write("### Limitations")
                for item in ai_result["limitations"]:
                    st.write("•", item)

            st.download_button(
                "Download AI analysis JSON",
                data=json.dumps(ai_result, indent=2, ensure_ascii=False),
                file_name=f'{st.session_state.case_id}_ai_analysis.json',
                mime="application/json",
                use_container_width=True,
            )

    # --------------------------------------------------------
    # DOWNLOAD ANALYSIS JSON
    # --------------------------------------------------------
    st.divider()
    phase2_export = build_phase2_export(analysis)
    st.download_button(
        "Download analysis.json",
        data=json.dumps(phase2_export, indent=2, ensure_ascii=False),
        file_name=f'{st.session_state.case_id}_analysis.json',
        mime="application/json",
        type="primary",
        use_container_width=True,
    )

    if st.button("← Back to analyzer", use_container_width=True):
        go("analyzer")

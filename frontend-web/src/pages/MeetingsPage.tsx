import { useEffect, useMemo, useState } from "react";
import { meetingsApi } from "../api/meetings";
import type { EmailDraft, Meeting, Minutes, Segment } from "../api/meetings";

export default function MeetingsPage() {
  const [items, setItems] = useState<Meeting[]>([]); const [selected, setSelected] = useState<Meeting>();
  const [segments, setSegments] = useState<Segment[]>([]); const [minutes, setMinutes] = useState<Minutes | null>(null);
  const [title, setTitle] = useState(""); const [participants, setParticipants] = useState("");
  const [file, setFile] = useState<File>(); const [audioUrl, setAudioUrl] = useState("");
  const [speakerNames, setSpeakerNames] = useState<Record<string, string>>({}); const [draft, setDraft] = useState<EmailDraft>();
  const [message, setMessage] = useState(""); const speakers = useMemo(() => [...new Set(segments.map(x => x.speaker_id))], [segments]);

  async function refreshList() { setItems((await meetingsApi.list()).items); }
  async function open(meeting: Meeting) {
    setSelected(await meetingsApi.get(meeting.id));
    const [transcript, latest] = await Promise.all([meetingsApi.transcript(meeting.id), meetingsApi.minutes(meeting.id)]);
    setSegments(transcript.segments); setMinutes(latest);
    setSpeakerNames(Object.fromEntries(transcript.segments.map(x => [x.speaker_id, x.speaker_name])));
    try { setAudioUrl(await meetingsApi.audio(meeting.id)); } catch { setAudioUrl(""); }
  }
  useEffect(() => { void refreshList(); }, []);
  useEffect(() => { if (!selected || !["transcribing", "generating_minutes"].includes(selected.status)) return; const timer = window.setInterval(async () => { const next = await meetingsApi.get(selected.id); setSelected(next); await refreshList(); if (!["transcribing", "generating_minutes"].includes(next.status)) await open(next); }, 1500); return () => clearInterval(timer); }, [selected?.id, selected?.status]);
  async function run(action: () => Promise<unknown>, done = "已完成") { try { setMessage("处理中…"); await action(); setMessage(done); await refreshList(); if (selected) await open(selected); } catch (error) { setMessage(error instanceof Error ? error.message : "操作失败"); } }

  return <div className="meeting-page">
    <div className="page-title"><h2>会议纪要 Agent</h2><p>真实转写、证据化纪要、人工审核、确认后发送</p></div>
    <div className="meeting-layout">
      <aside className="card meeting-list"><h3>会议</h3><div className="form">
        <input value={title} onChange={e => setTitle(e.target.value)} placeholder="会议标题" />
        <input value={participants} onChange={e => setParticipants(e.target.value)} placeholder="参会人（逗号分隔，可选）" />
        <button onClick={() => run(async () => { const x = await meetingsApi.create(title, participants.split(",").map(v => v.trim()).filter(Boolean)); setTitle(""); await open(x); }, "会议已创建")}>新建会议</button>
      </div><div className="meeting-items">{items.map(item => <button className={selected?.id === item.id ? "meeting-item active" : "meeting-item"} onClick={() => void open(item)} key={item.id}><strong>{item.title}</strong><span>{item.status} · {item.progress}%</span></button>)}</div></aside>
      <main className="meeting-workspace">{selected ? <>
        <section className="card"><div className="section-header"><div><h3>{selected.title}</h3><span className={selected.status === "failed" ? "status-failed" : "status-ok"}>{selected.status} · {selected.progress}%</span></div></div>
          {selected.error_message && <p className="error-text">{selected.error_message}</p>}<div className="meeting-actions"><input type="file" accept=".mp3,.wav,.m4a" onChange={e => setFile(e.target.files?.[0])}/><button disabled={!file} onClick={() => run(() => meetingsApi.upload(selected.id, file!), "音频已上传")}>上传音频</button><button onClick={() => run(() => meetingsApi.process(selected.id), "处理已启动")}>开始处理</button>{selected.status === "failed" && <button onClick={() => run(() => meetingsApi.retry(selected.id), "重试已启动")}>重试</button>}</div>{audioUrl && <audio controls src={audioUrl} />}</section>
        <div className="review-grid"><section className="card transcript-pane"><h3>转写与说话人</h3>{speakers.map(s => <label key={s}>{s}<input value={speakerNames[s] ?? s} onChange={e => setSpeakerNames({...speakerNames, [s]: e.target.value})}/></label>)}{speakers.length > 0 && <button onClick={() => run(() => meetingsApi.speakers(selected.id, speakers.map(s => ({speaker_id:s, display_name:speakerNames[s] || s, representative_segment_id:segments.find(x => x.speaker_id === s)?.id}))), "映射已保存")}>保存映射</button>}<div className="segments">{segments.map(s => <button className="segment" key={s.id} onClick={() => { const audio = document.querySelector("audio"); if (audio) { audio.currentTime = s.start_time; void audio.play(); }}}><strong>{s.speaker_name}</strong><time>{s.start_time.toFixed(1)}s</time><span>{s.text}</span></button>)}</div></section>
          <section className="card minutes-pane"><h3>结构化纪要</h3>{minutes ? <><label>摘要<textarea value={minutes.summary} onChange={e => setMinutes({...minutes, summary:e.target.value})}/></label><label>议题（JSON）<textarea value={JSON.stringify(minutes.topics, null, 2)} onChange={e => { try { setMinutes({...minutes, topics:JSON.parse(e.target.value)}); } catch {} }}/></label><label>决策与证据（JSON）<textarea value={JSON.stringify(minutes.decisions, null, 2)} onChange={e => { try { setMinutes({...minutes, decisions:JSON.parse(e.target.value)}); } catch {} }}/></label><label>行动项（JSON）<textarea value={JSON.stringify(minutes.action_items, null, 2)} onChange={e => { try { setMinutes({...minutes, action_items:JSON.parse(e.target.value)}); } catch {} }}/></label><div className="meeting-actions"><button onClick={() => run(() => meetingsApi.saveMinutes(selected.id, minutes), "纪要已保存，需重新审核")}>保存编辑</button><button onClick={() => run(() => meetingsApi.approve(selected.id), "当前版本已批准")}>批准版本</button><button onClick={() => run(async () => setDraft(await meetingsApi.createDraft(selected.id)), "邮件草稿已生成")}>生成邮件草稿</button></div></> : <p className="muted">处理完成后在此审核纪要。</p>}</section></div>
        {draft && <section className="card email-preview"><h3>邮件草稿（发送前需再次确认）</h3><label>To<input value={draft.to_addresses.join(",")} onChange={e => setDraft({...draft,to_addresses:e.target.value.split(",").map(x=>x.trim()).filter(Boolean)})}/></label><label>Cc<input value={draft.cc_addresses.join(",")} onChange={e => setDraft({...draft,cc_addresses:e.target.value.split(",").map(x=>x.trim()).filter(Boolean)})}/></label><label>Subject<input value={draft.subject} onChange={e => setDraft({...draft,subject:e.target.value})}/></label><label>Body<textarea value={draft.body} onChange={e => setDraft({...draft,body:e.target.value})}/></label><div className="meeting-actions"><button onClick={() => run(async () => setDraft(await meetingsApi.saveDraft(selected.id,draft)), "草稿已保存")}>保存草稿</button><button className="danger" onClick={() => run(() => meetingsApi.send(selected.id,draft.revision), "邮件已发送")}>明确确认并发送</button></div></section>}
      </> : <section className="card"><p className="muted">新建或选择会议开始。</p></section>}</main>
    </div>{message && <div className="meeting-toast">{message}</div>}</div>;
}

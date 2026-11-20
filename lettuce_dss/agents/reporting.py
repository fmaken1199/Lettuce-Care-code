"""An output decision agent with one bounded reconsideration request."""
from __future__ import annotations
import copy
import html
from lettuce_dss.core import BaseAgent, CLASS_NAMES, decision, utc_now
from lettuce_dss.context_display import context_label, environment_view, environment_html, environment_markdown, environment_summary, markdown_text
from lettuce_dss.crop_profile import crop_profile_rows


def action_sentence(action):
    """Use a complete instruction, never cut away a dose or a condition."""
    detail = " ".join(str(action.get('detail', '')).split())
    title = " ".join(str(action.get('title', 'Check the crop')).split())
    text = detail if detail and len(detail) <= 220 else title
    return text if text.endswith(('.', '!', '?')) else text + '.'


def medication_display(data):
    """Independent final gate, also applied to exported report records."""
    supplied = copy.deepcopy(data.get('medication') or {})
    prediction = data.get('prediction')
    if data.get('uncertainty') is not False or data.get('diagnosis_usable') is not True or not prediction:
        return {'status': 'not_selected', 'summary': 'No medication selected from this image result.', 'options': [], 'source_ids': []}
    if prediction == 'Healthy':
        return {'status': 'not_needed', 'summary': 'No disease medication is indicated by this image result.', 'options': [], 'source_ids': []}
    if prediction in {'Lettuce_Mosaic_Virus', 'Viral'}:
        return {'status': 'no_curative', 'summary': 'No curative medication is selected for a viral disease. Focus on limiting spread.',
                'options': [], 'source_ids': supplied.get('source_ids', []) if supplied.get('status') == 'no_curative' else []}
    if prediction in {'Bacterial', 'Wilt_and_Leaf_Blight'} or data.get('recommendation_scope') != 'disease_specific':
        return {'status': 'not_selected', 'summary': 'This broad disease pattern does not identify a medication target.', 'options': [], 'source_ids': []}
    if prediction not in {'Anthracnose', 'Bottom_Rot', 'Downy_Mildew', 'Powdery_Mildew', 'Septoria_Blight'}:
        return {'status': 'not_selected', 'summary': 'No medication selected for this image result.', 'options': [], 'source_ids': []}
    setup = data.get('farm_setup') or {}
    options = []
    for option in supplied.get('options', []):
        if not isinstance(option, dict) or not option.get('active_ingredient') or not option.get('source_ids'):
            continue
        if 'treatment_type' in option:
            # Check disease and cultivation evidence.
            if (option.get('treatment_type') not in {'chemical', 'biological'}
                or prediction not in option.get('diseases', [])
                or setup.get('environment') not in option.get('environments', [])
                or setup.get('root_system') not in option.get('root_systems', [])
                or not option.get('jurisdiction') or not option.get('review_status')
                or option.get('regulatory_status') != 'not_assessed_for_user_location'):
                continue
        elif setup.get('environment') != 'outdoor' or setup.get('root_system') != 'soil':
            # Do not widen the scope of older references.
            continue
        options.append(option)
    options = options[:2]
    if supplied.get('status') != 'options':
        options = []
    for option in options:
        # Do not display product doses or labels.
        for key in ('product_name', 'dose', 'phi', 'application_rate', 'rei'):
            option[key] = None
    return {'status': 'not_selected' if supplied.get('status') == 'options' and not options else supplied.get('status', 'unavailable'),
            'summary': (supplied.get('summary') if options or supplied.get('status') != 'options' else None) or 'No supported treatment reference is available for this result and growing setup.',
            'options': options,
            'source_ids': sorted(set(supplied.get('source_ids', [])) | {s for o in options for s in o.get('source_ids', [])})}


def medication_lines(medication):
    """Farmer-facing text shared by the screen and reports."""
    lines = []
    for option in medication.get('options', []):
        line = str(option['active_ingredient'])
        if option.get('purpose'):
            line += ': ' + str(option['purpose']).rstrip('.')
        line += '.'
        if option.get('farmer_note'):
            line += ' ' + str(option['farmer_note'])
        lines.append(line)
    scopes = list(dict.fromkeys(str(o['evidence_scope']) for o in medication.get('options', []) if o.get('evidence_scope')))
    return lines, scopes


def medication_use_note(medication):
    if not medication.get('options'):
        return ''
    return ("Before use, confirm that the product is approved locally for lettuce and this disease. "
            "Follow the label's dose and waiting period before harvest. "
            "If unsure, consult an agricultural adviser.")


def medication_evidence_lines(option):
    fields = (('application_note', 'Application reference'), ('evidence_scope', 'Evidence scope'),
              ('jurisdiction', 'Evidence country'), ('regulatory_status', 'Local authorization status'),
              ('review_status', 'Review status'))
    return [label + ': ' + str(option[key]) for key, label in fields if option.get(key)]


def evidence_sources(item, sources):
    """Resolve per-item evidence without trusting arbitrary URL schemes."""
    lookup = sources if isinstance(sources, dict) else {s['id']: s for s in sources}
    embedded = {s['id']: s for s in item.get('sources', []) if isinstance(s, dict) and s.get('id')}
    rows = []
    for sid in item.get('source_ids', []):
        source = lookup.get(sid) or embedded.get(sid) or {}
        url = str(source.get('url', ''))
        rows.append({'id': sid, 'title': str(source.get('title', sid)),
                     'url': url if url.startswith(('https://', 'http://')) else ''})
    return rows


def item_trace_text(item):
    reason = item.get('reason') or item.get('purpose') or 'Selection reason not recorded.'
    return 'Reason: ' + str(reason) + ' Rules: ' + (', '.join(item.get('triggered_rule_ids', [])) or 'Not recorded') + \
        '. Sources: ' + (', '.join(item.get('source_ids', [])) or 'Not recorded') + '.'


def audit_lines(data):
    lines = recommendation_usage_lines(data)
    kb = data.get('knowledge_base') or {}
    if kb:
        lines.append('Knowledge base version: ' + str(kb.get('version', 'Not recorded')))
        lines.append('Knowledge base SHA-256: ' + str(kb.get('sha256', 'Not recorded')))
        lines.append('Knowledge base counts: ' + '; '.join(str(k) + '=' + str(v) for k, v in kb.get('counts', {}).items()))
    for item in data.get('rule_trace', []):
        if isinstance(item, dict):
            lines.append(str(item.get('status', 'Not recorded')).capitalize() + ': ' +
                         str(item.get('item_id', 'Item')) + ' (' + str(item.get('item_type', '')) + '). ' +
                         str(item.get('reason', '')) + ' Rules: ' + ', '.join(item.get('rule_ids', [])) +
                         '. Sources: ' + ', '.join(item.get('source_ids', [])) + '.')
    return lines


def recommendation_usage_lines(data):
    """Separate this run's recorded decisions/evidence from registry size."""
    rule_ids = sorted({rule_id for rule_id in data.get('triggered_rule_ids', []) if isinstance(rule_id, str)})
    source_ids = sorted({source_id for source_id in data.get('source_ids', []) if isinstance(source_id, str)})
    rule_line = ('Rules recorded in this recommendation run: ' + str(len(rule_ids)) +
                 (' (' + ', '.join(rule_ids) + ').' if rule_ids else '.')) if 'triggered_rule_ids' in data else 'Rule usage was not recorded for this report.'
    return [rule_line,
            'Distinct sources linked to selected recommendations: ' + str(len(source_ids)) +
            ' (all selected care actions and treatment references, including actions retained in the details).',
            'Sources for excluded or deferred options are listed separately in the decision trace.']


def _reference_ids(data, medication):
    return sorted(set(data.get('source_ids', [])) | set(medication.get('source_ids', [])) |
                  {sid for row in data.get('rule_trace', []) for sid in row.get('source_ids', [])})


def _evidence_markdown(item, sources):
    links = []
    for row in evidence_sources(item, sources):
        label = markdown_text(row['id'] + ': ' + row['title'])
        url = row['url'].replace(' ', '%20').replace('(', '%28').replace(')', '%29')
        links.append('[' + label + '](' + url + ')' if url else label)
    return '; '.join(links)


def _evidence_html(item, sources):
    links = []
    for row in evidence_sources(item, sources):
        label = html.escape(row['id'] + ': ' + row['title'])
        links.append('<a href="' + html.escape(row['url'], quote=True) + '">' + label + '</a>' if row['url'] else label)
    return '; '.join(links)

class ReportingAgent(BaseAgent):
    name = "reporting"
    goal = "Return a concise, internally consistent account of the available advice."

    def run(self, results: dict, farm: dict, feedback_used: bool = False):
        self.begin()
        diag_result = results.get('diagnosis',{})
        diag = diag_result.get('data',{})
        env_result = results.get('environment',{})
        env = env_result.get('data',{})
        rec_result = results.get('recommendation',{})
        rec = rec_result.get('data',{})
        needs_image = results.get('image_quality',{}).get('status') == 'blocked'
        diagnosis_error = diag_result.get('status') not in {'ok','partial'} or diag.get('prediction') not in CLASS_NAMES
        uncertain = diag.get('uncertainty') is not False
        scope = 'disease_specific' if (not needs_image and not diagnosis_error and not uncertain
            and diag.get('recommendation_scope') == 'disease_specific'
            and rec.get('recommendation_scope', 'disease_specific') == 'disease_specific') else 'general_only'
        recommendation_available = rec_result.get('status') in {'ok', 'partial'}
        actions = copy.deepcopy(rec.get('actions',[])) if recommendation_available else []
        decisions = []
        contradictions = []
        for a in actions:
            # Recheck incompatible growing conditions.
            allowed = a.get('root_systems')
            if allowed and farm.get('root_system') not in allowed:
                contradictions.append(a['id'])
            allowed_environments = a.get('environments')
            if allowed_environments and farm.get('environment') not in allowed_environments:
                contradictions.append(a['id'])
            if scope == 'general_only' and a.get('requires_specific_diagnosis',False):
                contradictions.append(a['id'])
        request = None
        if contradictions and not feedback_used:
            request = {'kind':'request', 'sender':self.name,'recipient':'recommendation',
                       'payload':{'exclude_ids':sorted(set(contradictions)),
                                  'reason':'Action context conflicts with farm setup or diagnostic scope.'}}
            decisions.append(decision('request_reconsideration','Context-conflicting actions were returned.', action_ids=contradictions))
        # Request missing context once.
        elif not env_result and not feedback_used:
            request = {'kind':'request','sender':self.name,'recipient':'environment',
                       'payload':{'action':'summarize_missing'}}
            decisions.append(decision('request_missing_result','Environmental status has not been supplied.'))
        if contradictions:
            actions = [a for a in actions if a['id'] not in contradictions]
        actions.sort(key=lambda a: (a.get('priority',99),a.get('id','')))
        chosen = actions[:4]
        decisions.append(decision('prioritize_report','Display up to four actions; retain all eligible actions in the technical record.',shown=len(chosen),available=len(actions)))
        if needs_image:
            headline = 'Please upload a clearer lettuce-leaf photograph.'
        elif diagnosis_error:
            headline = 'The image could not be assessed in this run.'
        elif not isinstance(diag.get('uncertainty'), bool):
            headline = 'The image confidence assessment is unavailable. Only general care actions are shown.'
        elif uncertain:
            headline = 'The image result is uncertain. Start with the general care actions below.'
        elif diag.get('prediction') == 'Healthy':
            headline = 'The image was classified as healthy.'
        else:
            headline = 'Likely condition: ' + str(diag.get('prediction','Unavailable')).replace('_',' ')
        medication = medication_display({'prediction': diag.get('prediction'), 'uncertainty': uncertain,
            'diagnosis_usable': not (needs_image or diagnosis_error), 'recommendation_scope': scope,
            'farm_setup': {'environment': farm.get('environment'), 'root_system': farm.get('root_system')},
            'medication': rec.get('medication', {}) if recommendation_available else {}})
        if rec.get('medication', {}).get('options') and not medication['options']:
            decisions.append(decision('withhold_medication', 'The diagnostic scope does not support the supplied medication options.'))
        data = {'headline':headline,'prediction':diag.get('prediction'),
                'confidence':diag.get('confidence'),'uncertainty':uncertain,
                'diagnosis_usable': not (needs_image or diagnosis_error), 'recommendation_scope': scope,
                'diagnosis_model': {key: copy.deepcopy(diag[key]) for key in (
                    'model_id', 'backend', 'display_name', 'model_mode', 'calibration_status',
                    'probability_kind', 'temperature', 'threshold_status') if key in diag},
                'recommendation_status':rec_result.get('status','missing'),
                'recommendation_notice': '' if recommendation_available else 'Treatment recommendations could not be completed for this result.',
                'actions':chosen,'all_actions':actions,'summary':rec.get('summary',''),
                'medication': medication,
                'source_ids':sorted({s for a in actions for s in a.get('source_ids',[])} | set(medication['source_ids'])),
                'environment_summary':env,
                'farm_setup': {**{key:farm.get(key) for key in ('environment','root_system','growth_stage','irrigation')},
                               'input_provenance':copy.deepcopy(farm.get('input_provenance',{}))},
                'input_provenance':copy.deepcopy(farm.get('input_provenance',{})),
                'assumed_context':copy.deepcopy(rec.get('assumed_context',[])),
                'knowledge_base':copy.deepcopy(rec.get('knowledge_base',{})),
                'knowledge_version':rec.get('knowledge_version'), 'knowledge_sha256':rec.get('knowledge_sha256'),
                'knowledge_counts':copy.deepcopy(rec.get('knowledge_counts',{})),
                'triggered_rule_ids':copy.deepcopy(rec.get('triggered_rule_ids',[])),
                'rule_trace':copy.deepcopy(rec.get('rule_trace',[])) if recommendation_available else [],
                'excluded_actions':copy.deepcopy(rec.get('excluded_actions',[])) if recommendation_available else [],
                'deferred_actions':copy.deepcopy(rec.get('deferred_actions',[])) if recommendation_available else [],
                'location':farm.get('location',{}),'missing_context':list(dict.fromkeys(env.get('missing',[])+rec.get('missing_context',[]))),
                'generated_at':utc_now(),'feedback_request':request,
                'agent_trace': [{'agent': name, 'status': result.get('status'), 'decisions': result.get('decisions', [])}
                                for name, result in results.items() if isinstance(result, dict)]}
        status = 'blocked' if needs_image or diagnosis_error else ('partial' if uncertain
            or diag_result.get('status') != 'ok' or env_result.get('status') != 'ok'
            or rec_result.get('status') != 'ok' or contradictions else 'ok')
        data['agent_trace'].append({'agent': self.name, 'status': status, 'decisions': copy.deepcopy(decisions)})
        return self.result(data,status,decisions,[request] if request else [])

def report_markdown(report: dict, sources: dict) -> str:
    d = report.get('data',report)
    med = medication_display(d)
    med_lines, _ = medication_lines(med)
    lines = ['# Lettuce Care Pakistan', '', markdown_text(d.get('headline','Report')), '',
             'Generated: ' + markdown_text(d.get('generated_at','')), '']
    if d.get('recommendation_notice'):lines += [markdown_text(d['recommendation_notice']), '']
    lines += ['## Recommended actions', '']
    lines += ['- ' + markdown_text(action_sentence(a)) for a in d.get('actions', [])[:4]] or ['No actions selected.']
    lines += ['', '## Possible treatment options', '', markdown_text(med['summary']), '']
    lines += ['- ' + markdown_text(line) for line in med_lines]
    if medication_use_note(med): lines += ['', markdown_text(medication_use_note(med))]
    lines += ['', '## Soil/weather summary', '']
    lines += [markdown_text(line) + '  ' for line in environment_summary(d.get('environment_summary', {}), d.get('farm_setup'))]
    lines += ['', '<details>', '<summary>Evidence and details</summary>', '']
    model = d.get('diagnosis_model', {})
    if model:
        lines += ['Image model: ' + markdown_text(model.get('display_name') or model.get('model_id', 'Unavailable')),
                  'Mode: ' + markdown_text(model.get('model_mode', 'Unavailable')),
                  'Calibration: ' + markdown_text(str(model.get('calibration_status', 'Unavailable')).replace('_', ' ')), '']
    missing=d.get('missing_context',[])
    if missing:lines += ['Unavailable context: '+', '.join(markdown_text(context_label(k)) for k in missing), '']
    lines += ['### Crop settings and assumptions', '']
    for row in crop_profile_rows(d.get('farm_setup',{})):
        lines += ['- ' + markdown_text(row['Field'] + ': ' + row['Setting'] + ' (' + row['Origin'] + ').')]
    lines += ['Default assumptions are initial form settings, not confirmed observations.', '']
    lines += [environment_markdown(environment_view(d.get('environment_summary',{}),d.get('farm_setup'),d.get('location'))), '']
    lines += ['### Action details', '']
    for action in d.get('all_actions', d.get('actions', [])):
        lines += ['- ' + markdown_text(action.get('title', 'Action')) + ': ' + markdown_text(action.get('detail', '')) +
                  ' ' + markdown_text(item_trace_text(action)), '  Evidence: ' + _evidence_markdown(action, sources)]
    lines += ['', '### Treatment reference details', '']
    for option in med['options']:
        lines += ['- ' + markdown_text(option['active_ingredient'] + '. ' + item_trace_text(option)),
                  '  Evidence: ' + _evidence_markdown(option, sources)]
        lines += ['  ' + markdown_text(line) + '  ' for line in medication_evidence_lines(option)]
    if audit_lines(d):
        lines += ['', '### Knowledge base and decision trace', '']
        lines += ['- ' + markdown_text(line) for line in audit_lines(d)]
    lines += ['', '### Sources', '']
    lookup = sources if isinstance(sources,dict) else {s['id']:s for s in sources}
    for sid in _reference_ids(d, med):
        s=lookup.get(sid,{})
        if isinstance(s,dict):
            url = str(s.get('url',''))
            if url.startswith(('https://','http://')):
                lines.append('- ' + markdown_text(sid + ': ' + str(s.get('title',sid))) + ': ' + markdown_text(url))
    if d.get('agent_trace'):
        lines += ['', '### Agent decisions', '']
        for item in d['agent_trace']:
            text = str(item.get('agent', 'Agent')) + ': ' + str(item.get('status', 'unknown')) + '. '
            text += '; '.join(str(dec.get('reason') or dec.get('action', '')) for dec in item.get('decisions', []))
            lines += ['- ' + markdown_text(text)]
    return '\n'.join(lines + ['', '</details>', ''])

def report_html(report: dict, sources: dict) -> str:
    d=report.get('data',report);e=html.escape
    med=medication_display(d); med_lines, _=medication_lines(med)
    actions=''.join('<li>'+e(action_sentence(a))+'</li>' for a in d.get('actions',[])[:4])
    cards=('<p>'+e(str(d['recommendation_notice']))+'</p>') if d.get('recommendation_notice') else ''
    cards+='<section id="recommended-actions"><h2>Recommended actions</h2>'+('<ul>'+actions+'</ul>' if actions else '<p>No actions selected.</p>')+'</section>'
    cards+='<section id="medication"><h2>Possible treatment options</h2><p>'+e(str(med['summary']))+'</p>'
    if med_lines: cards+='<ul>'+''.join('<li>'+e(line)+'</li>' for line in med_lines)+'</ul>'
    if medication_use_note(med): cards+='<p>'+e(medication_use_note(med))+'</p>'
    cards+='</section><section id="soil-weather-summary"><h2>Soil/weather summary</h2>'
    cards+=''.join('<p>'+e(line)+'</p>' for line in environment_summary(d.get('environment_summary',{}),d.get('farm_setup')))+'</section>'
    lookup=sources if isinstance(sources,dict) else {s['id']:s for s in sources}
    refs=[]
    for sid in _reference_ids(d, med):
        s=lookup.get(sid,{})
        if isinstance(s,dict):
            u=str(s.get('url',''));title=e(sid + ': ' + str(s.get('title',sid)))
            if u.startswith(('https://','http://')):refs.append('<li><a href="'+e(u,quote=True)+'">'+title+'</a></li>')
    missing=', '.join(context_label(k) for k in d.get('missing_context',[]))
    context=environment_html(environment_view(d.get('environment_summary',{}),d.get('farm_setup'),d.get('location')))
    details='<details><summary>Evidence and details</summary>'
    model=d.get('diagnosis_model',{})
    if model:
        details+='<p>Image model: '+e(str(model.get('display_name') or model.get('model_id','Unavailable')))+\
            '<br>Mode: '+e(str(model.get('model_mode','Unavailable')))+\
            '<br>Calibration: '+e(str(model.get('calibration_status','Unavailable')).replace('_',' '))+'</p>'
    if missing: details+='<p>Unavailable context: '+e(missing)+'</p>'
    details+='<h3>Crop settings and assumptions</h3><ul>'
    details+=''.join('<li>'+e(row['Field']+': '+row['Setting']+' ('+row['Origin']+').')+'</li>' for row in crop_profile_rows(d.get('farm_setup',{})))+'</ul>'
    details+='<p>Default assumptions are initial form settings, not confirmed observations.</p>'
    details+=context+'<h3>Action details</h3><ul>'
    details+=''.join('<li><strong>'+e(str(a.get('title','Action')))+'</strong>: '+e(str(a.get('detail','')))+'<br>'+e(item_trace_text(a))+'<br>Evidence: '+_evidence_html(a,sources)+'</li>' for a in d.get('all_actions',d.get('actions',[])))+'</ul>'
    details+='<h3>Treatment reference details</h3><ul>'
    for option in med['options']:
        details+='<li><strong>'+e(str(option['active_ingredient']))+'</strong><br>'+e(item_trace_text(option))+'<br>Evidence: '+_evidence_html(option,sources)
        details+=''.join('<br>'+e(line) for line in medication_evidence_lines(option))
        details+='</li>'
    details+='</ul>'
    if audit_lines(d):
        details+='<h3>Knowledge base and decision trace</h3><ul>'+''.join('<li>'+e(line)+'</li>' for line in audit_lines(d))+'</ul>'
    details+='<h3>Sources</h3><ul>'+''.join(refs)+'</ul>'
    if d.get('agent_trace'):
        details+='<h3>Agent decisions</h3><ul>'
        for item in d['agent_trace']:
            text=str(item.get('agent','Agent'))+': '+str(item.get('status','unknown'))+'. '
            text+='; '.join(str(dec.get('reason') or dec.get('action','')) for dec in item.get('decisions',[]))
            details+='<li>'+e(text)+'</li>'
        details+='</ul>'
    details+='</details>'
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lettuce Care Pakistan report</title><style>body{font:16px/1.55 system-ui,sans-serif;max-width:1000px;margin:32px auto;padding:0 24px;color:#21382a}h1{color:#245c39}section{border:1px solid #dce6da;border-radius:12px;margin:16px 0;padding:14px 20px}section p{margin:8px 0}h2{font-size:19px;margin-top:0}h3{font-size:17px}.evidence,small{color:#526652}a{color:#286742}li{margin:8px 0}summary{cursor:pointer;font-weight:600;padding:12px 0}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;vertical-align:top;border:1px solid #dce6da;padding:9px;overflow-wrap:anywhere}th{background:#f2f6ee}td:last-child{min-width:160px}@media print{body{margin:12px;max-width:none}.table-wrap{overflow:visible}thead{display:table-header-group}tr{break-inside:avoid}table{font-size:11px}section{break-inside:avoid}}</style></head><body><h1>Lettuce Care Pakistan</h1><p>'+e(str(d.get('headline','')))+'<br><small>'+e(str(d.get('generated_at','')))+'</small></p>'+cards+details+'</body></html>'
